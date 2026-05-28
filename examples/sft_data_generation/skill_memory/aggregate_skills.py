"""
Aggregate raw per-trajectory memories into a polished skill bank.

The output schema mirrors `skill_generation/{alfworld,webshop,search}.py` so
the same JSON can feed both the RL runtime (`SkillsOnlyMemory`) and the SFT
distillation pipeline.

Output schema:
    {
      "general_skills": [
        {"skill_id": "gen_001",
         "title": "Systematic Exploration",
         "principle": "Search every plausible ...",
         "when_to_apply": "When starting any task ..."},
        ...
      ],
      "task_specific_skills": {                # alfworld + webshop
        "<category>": [
          {"skill_id": "<prefix>_001",
           "title": "...",
           "principle": "...",
           "when_to_apply": "..."},
          ...
        ],
        ...
      },
      "query_type_skills": {...},              # search only (matches skill_generation/search.py)
      "common_mistakes": [
        {"mistake_id": "err_001",
         "description": "...",
         "why_it_happens": "...",
         "how_to_avoid": "..."},
        ...
      ]
    }

Usage:
    export OPENAI_API_KEY=...
    python aggregate_skills.py \\
        --input_file generated_memories_alfworld.json \\
        --output_file alfworld_skill_bank.json \\
        --env alfworld \\
        --model gpt-4o
"""
import argparse
import json
import os
import re

from openai import OpenAI

# ---------------------------------------------------------------------------
# Category sets — kept aligned with `skill_generation/{alfworld,webshop,search}.py`
# ---------------------------------------------------------------------------

ALFWORLD_CATEGORIES = [
    "pick_and_place",
    "look_at_obj_in_light",
    "clean",
    "heat",
    "cool",
    "examine",
]

ALFWORLD_CATEGORY_DISPLAY = {
    "pick_and_place": "Pick And Place",
    "look_at_obj_in_light": "Look At Object In Light",
    "clean": "Clean",
    "heat": "Heat",
    "cool": "Cool",
    "examine": "Examine",
}

WEBSHOP_CATEGORIES = [
    "apparel",
    "footwear",
    "home_decor",
    "electronics",
    "accessories",
    "beauty_health",
    "other",
]

SEARCH_CATEGORIES = [
    "direct_retrieval",
    "multi_hop_reasoning",
    "entity_attribute_lookup",
    "comparison",
]


# ---------------------------------------------------------------------------
# Prompts — instruct the LLM to emit the skill_generation/ field schema
# ---------------------------------------------------------------------------

GENERAL_PRINCIPLES_PROMPT = """You are an expert agent-memory summarizer.

You will be given a list of `planning_pattern` strings extracted from successful agent trajectories. Each is a short action chain template like:
    "Search [Location] -> Acquire [Object] -> Navigate to [Target_Location] -> Place [Object]"

Your job: produce 5-7 high-level GENERAL SKILLS that capture the common,
universally-applicable strategies behind these patterns. Each skill should
be polished into clear English prose, ready to drop into a system prompt.

Output format (strict JSON list, no preamble, no markdown fences):
[
  {
    "title": "<Title Case short name, 3-5 words>",
    "principle": "<one or two sentences ending with period; the core actionable insight>",
    "when_to_apply": "<one sentence ending with period; the trigger condition>"
  },
  ...
]

Constraints:
- DO NOT use bracketed placeholders like [Object] or [Location] in the output. Write natural English.
- 5 to 7 entries total.
- Each entry should be self-contained and actionable."""


CATEGORY_SKILLS_PROMPT = """You are an expert agent-memory summarizer.

You will be given a list of `planning_pattern` strings from successful trajectories of one specific task category: **{category_label}**. Each is a short action chain template.

Your job: produce 5-6 CATEGORY-SPECIFIC SKILLS that capture the
distinctive techniques for this task type.

Output format (strict JSON list, no preamble, no markdown fences):
[
  {{
    "title": "<Title Case short name, 3-5 words>",
    "principle": "<one or two sentences ending with period; the actionable insight specific to this task category>",
    "when_to_apply": "<one sentence ending with period; the trigger condition>"
  }},
  ...
]

Constraints:
- DO NOT use bracketed placeholders. Write natural English.
- Skills should be specific to the {category_label} task type, not generic.
- 5 to 6 entries total."""


MISTAKES_PROMPT = """You are an expert agent-memory summarizer.

You will be given a list of `mistakes_to_avoid` items extracted from failed agent trajectories. Each item has a `trigger_condition` and a `bad_action`.

Your job: produce 5 COMMON MISTAKES entries that capture the most common,
generalizable failure modes across these failed trajectories. Dedupe
similar mistakes and polish into clear English.

Output format (strict JSON list, no preamble, no markdown fences):
[
  {
    "description": "<one sentence describing the bad behavior>",
    "why_it_happens": "<one sentence explaining why agents make this mistake>",
    "how_to_avoid": "<one or two sentences describing the concrete correction>"
  },
  ...
]

Constraints:
- DO NOT use bracketed placeholders. Write natural English.
- Exactly 5 entries."""


# ---------------------------------------------------------------------------
# LLM call + JSON parsing
# ---------------------------------------------------------------------------

def call_llm(client: OpenAI, model: str, system_prompt: str, user_payload: dict) -> list:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        temperature=0,
    )
    text = resp.choices[0].message.content
    text = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise ValueError(f"LLM did not return valid JSON: {text[:300]}")


# ---------------------------------------------------------------------------
# Classifiers (per-env). Map a memory record to one of the category keys.
# ---------------------------------------------------------------------------

def classify_alfworld_memory(mem: dict) -> str | None:
    """Map a memory to one of ALFWORLD_CATEGORIES."""
    goal = (mem.get("content", {}).get("task_meta", {}).get("original_goal") or "").lower()
    if "look at" in goal and "under" in goal:
        return "look_at_obj_in_light"
    if "clean" in goal:
        return "clean"
    if "heat" in goal or "hot" in goal:
        return "heat"
    if "cool" in goal or "cold" in goal:
        return "cool"
    if "examine" in goal or "find" in goal:
        return "examine"
    if "put" in goal or "place" in goal:
        return "pick_and_place"
    desc = (mem.get("contextual_description") or "").lower()
    if "look_at_obj_in_light" in desc:
        return "look_at_obj_in_light"
    if "clean" in desc:
        return "clean"
    if "heat" in desc:
        return "heat"
    if "cool" in desc:
        return "cool"
    if "examine" in desc:
        return "examine"
    if "pick_and_place" in desc or "pick_two_obj_and_place" in desc:
        return "pick_and_place"
    return None


WEBSHOP_KEYWORD_TABLE = [
    ("apparel", [
        "shirt", "dress", "t-shirt", "polo", "pants", "jeans", "jacket", "coat",
        "sweater", "blouse", "skirt", "shorts", "underwear", "swimsuit", "swimwear",
        "hoodie", "vest", "cardigan", "suit", "blazer", "tee", "top",
    ]),
    ("footwear", [
        "shoe", "boot", "sandal", "sneaker", "slipper", "loafer", "heel", "flat",
        "oxford", "pump", "moccasin", "flip-flop", "footwear",
    ]),
    ("home_decor", [
        "pillow", "curtain", "rug", "mat", "blanket", "bedding", "towel", "lamp",
        "decor", "furniture", "cushion", "sheet", "tablecloth", "vase",
    ]),
    ("electronics", [
        "phone", "laptop", "tablet", "computer", "headphone", "earphone", "earbud",
        "speaker", "charger", "cable", "mouse", "keyboard", "monitor", "camera",
        "smartwatch", "battery", "electronic", "device", "gadget",
    ]),
    ("accessories", [
        "bag", "wallet", "belt", "hat", "cap", "scarf", "glove", "jewelry",
        "necklace", "bracelet", "ring", "earring", "sunglasses", "glasses", "watch",
        "purse", "backpack", "handbag", "tie", "bow",
    ]),
    ("beauty_health", [
        "makeup", "cosmetic", "skincare", "lotion", "cream", "shampoo", "conditioner",
        "perfume", "cologne", "brush", "bathing", "soap", "body wash", "nail",
        "lipstick", "mascara", "foundation", "serum", "moisturizer",
    ]),
]


def classify_webshop_memory(mem: dict) -> str:
    """Map a memory to one of WEBSHOP_CATEGORIES."""
    goal = (mem.get("content", {}).get("task_meta", {}).get("original_goal") or "").lower()
    for category, keywords in WEBSHOP_KEYWORD_TABLE:
        if any(kw in goal for kw in keywords):
            return category
    return "other"


def classify_search_memory(mem: dict) -> str:
    """Map a memory to one of SEARCH_CATEGORIES."""
    data_source = mem.get("tags", {}).get("data_source", "")
    goal = (mem.get("content", {}).get("task_meta", {}).get("original_goal") or "").lower()
    if data_source in ("hotpotqa", "2wikimultihopqa", "musique", "bamboogle"):
        comparison_kw = [
            "both", "are the", "which of", "same", "common", "more", "less",
            "older", "younger", "taller", "shorter",
        ]
        if any(kw in goal for kw in comparison_kw):
            return "comparison"
        return "multi_hop_reasoning"
    if data_source == "popqa":
        return "entity_attribute_lookup"
    return "direct_retrieval"


# ---------------------------------------------------------------------------
# Skill / mistake ID assignment — done after LLM call so IDs follow a clean
# deterministic scheme regardless of what the LLM emits.
# ---------------------------------------------------------------------------

def _id_prefix(category: str) -> str:
    """Match `skill_generation/`'s convention: first three letters of the category."""
    return re.sub(r"[^a-z]", "", category.lower())[:3] or "cat"


def _assign_skill_ids(skills: list, prefix: str) -> list:
    out = []
    for i, sk in enumerate(skills, 1):
        if not isinstance(sk, dict):
            continue
        sk = {k: v for k, v in sk.items() if k != "skill_id"}
        out.append({"skill_id": f"{prefix}_{i:03d}", **sk})
    return out


def _assign_mistake_ids(mistakes: list) -> list:
    out = []
    for i, m in enumerate(mistakes, 1):
        if not isinstance(m, dict):
            continue
        m = {k: v for k, v in m.items() if k != "mistake_id"}
        out.append({"mistake_id": f"err_{i:03d}", **m})
    return out


# ---------------------------------------------------------------------------
# Aggregation pipeline
# ---------------------------------------------------------------------------

ENV_CONFIG = {
    "alfworld": {
        "categories": ALFWORLD_CATEGORIES,
        "classifier": classify_alfworld_memory,
        "category_key": "task_specific_skills",
        "label_map": ALFWORLD_CATEGORY_DISPLAY,
    },
    "webshop": {
        "categories": WEBSHOP_CATEGORIES,
        "classifier": classify_webshop_memory,
        "category_key": "task_specific_skills",
        "label_map": None,
    },
    "search": {
        "categories": SEARCH_CATEGORIES,
        "classifier": classify_search_memory,
        "category_key": "query_type_skills",
        "label_map": None,
    },
}


def aggregate(memories: list[dict], env: str, client: OpenAI, model: str) -> dict:
    cfg = ENV_CONFIG[env]
    successes = [m for m in memories if m.get("tags", {}).get("outcome") == "Success"]
    failures = [m for m in memories if m.get("tags", {}).get("outcome") == "Failure"]

    # ---- General skills ------------------------------------------------
    print(f"Aggregating General Skills from {len(successes)} success memories...")
    all_patterns = [
        m.get("content", {}).get("strategic_guidelines", {}).get("planning_pattern", "")
        for m in successes
    ]
    all_patterns = [p for p in all_patterns if p]
    general = call_llm(client, model, GENERAL_PRINCIPLES_PROMPT, {"planning_patterns": all_patterns})
    general = _assign_skill_ids(general, prefix="gen")

    # ---- Per-category skills -------------------------------------------
    by_cat: dict[str, list[str]] = {c: [] for c in cfg["categories"]}
    for m in successes:
        cat = cfg["classifier"](m)
        if cat is None or cat not in by_cat:
            continue
        pat = m.get("content", {}).get("strategic_guidelines", {}).get("planning_pattern", "")
        if pat:
            by_cat[cat].append(pat)

    category_skills: dict[str, list] = {}
    for cat in cfg["categories"]:
        patterns = by_cat[cat]
        if not patterns:
            print(f"  [{cat}] no memories, skipping")
            category_skills[cat] = []
            continue
        if cfg["label_map"]:
            label = cfg["label_map"].get(cat, cat.replace("_", " ").title())
        else:
            label = cat.replace("_", " ").title()
        print(f"  [{cat}] aggregating from {len(patterns)} memories...")
        prompt = CATEGORY_SKILLS_PROMPT.format(category_label=label)
        skills = call_llm(client, model, prompt, {"planning_patterns": patterns})
        category_skills[cat] = _assign_skill_ids(skills, prefix=_id_prefix(cat))

    # ---- Common mistakes -----------------------------------------------
    print(f"Aggregating Common Mistakes from {len(failures)} failure memories...")
    all_mistakes: list = []
    for m in failures:
        sg = m.get("content", {}).get("strategic_guidelines", {}) or {}
        for mis in sg.get("mistakes_to_avoid", []) or []:
            all_mistakes.append(mis)
    if all_mistakes:
        mistakes = call_llm(client, model, MISTAKES_PROMPT, {"mistakes": all_mistakes})
        mistakes = _assign_mistake_ids(mistakes)
    else:
        mistakes = []

    return {
        "general_skills": general,
        cfg["category_key"]: category_skills,
        "common_mistakes": mistakes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="generated_memories_*.json from stage 3")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--env", choices=["alfworld", "webshop", "search"], required=True)
    parser.add_argument("--model", default="deepseek-v4-pro")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY in the environment.")
    # DeepSeek API 兼容 OpenAI SDK；保留环境变量覆盖，便于临时切回其它兼容端点。
    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )

    with open(args.input_file, "r", encoding="utf-8") as f:
        memories = json.load(f)

    skill_bank = aggregate(memories, args.env, client, args.model)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(skill_bank, f, indent=2, ensure_ascii=False)

    cat_key = ENV_CONFIG[args.env]["category_key"]
    print(f"\nWrote skill bank to {args.output_file}")
    print(f"  general_skills: {len(skill_bank['general_skills'])} items")
    for cat, items in skill_bank[cat_key].items():
        print(f"  {cat_key}[{cat}]: {len(items)} items")
    print(f"  common_mistakes: {len(skill_bank['common_mistakes'])} items")


if __name__ == "__main__":
    main()
