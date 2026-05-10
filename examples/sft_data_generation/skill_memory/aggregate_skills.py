"""
Aggregate raw per-trajectory memories into a polished skill bank.

The released SFT data shows that the skill block in each system prompt has
three sections, two of which are STATIC across all tasks of an env:

    ### General Principles      (static — same for every task)
    ### <Category> Skills       (varies by task category — pick_and_place, clean, ...)
    ### Mistakes to Avoid       (static — same for every task)

So we don't actually do per-trajectory retrieval. We aggregate once: take
all success memories, ask an LLM to summarize their planning patterns into
6-or-so polished principle bullets. Group success memories by task
category and ask the LLM to summarize each group into 5-6 category-specific
skill bullets. Take all failure memories' `mistakes_to_avoid` lists and
ask the LLM to dedupe + polish into ~5 don't/instead pairs.

The output is a single JSON skill bank that downstream distillation just
indexes by category.

Output schema:
    {
      "general_principles": [
        {"name": "Systematic Exploration", "description": "Search every plausible ..."}
      ],
      "category_skills": {
        "pick_and_place": [
          {"name": "Systematic First-Pass Search",
           "description": "Maintain a checklist of all visible ...",
           "apply_when": "After reading the goal and before acquiring ..."}
        ],
        "pick_clean_then_place_in_recep": [...],
        ...
      },
      "mistakes_to_avoid": [
        {"dont": "Agent repeatedly revisits ...",
         "instead": "Maintain an exploration map ..."}
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

ALFWORLD_CATEGORIES = [
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]

ALFWORLD_CATEGORY_DISPLAY = {
    "pick_and_place": "Pick And Place",
    "pick_two_obj_and_place": "Pick And Place",
    "look_at_obj_in_light": "Examine",
    "pick_heat_then_place_in_recep": "Heat",
    "pick_cool_then_place_in_recep": "Cool",
    "pick_clean_then_place_in_recep": "Clean",
}

WEBSHOP_CATEGORIES = ["general", "apparel", "electronics"]

SEARCH_CATEGORIES = ["direct_retrieval", "multi_hop"]

SEARCH_CATEGORY_DISPLAY = {
    "direct_retrieval": "Direct Retrieval",
    "multi_hop": "Multi-Hop Reasoning",
}


GENERAL_PRINCIPLES_PROMPT = """You are an expert agent-memory summarizer.

You will be given a list of `planning_pattern` strings extracted from successful agent trajectories. Each is a short action chain template like:
    "Search [Location] -> Acquire [Object] -> Navigate to [Target_Location] -> Place [Object]"

Your job: produce 5-7 high-level **General Principles** that are the
common, universally-applicable strategies behind these patterns. Each
principle should be a single short bullet polished into clear English
prose, ready to drop into a system prompt.

Output format (strict JSON list, no preamble, no markdown fences):
[
  {
    "name": "<Title Case Short Name, 2-3 words>",
    "description": "<one sentence in clear English, ending with period; describes the principle in general terms applicable across task types>"
  },
  ...
]

Constraints:
- DO NOT use bracketed placeholders like [Object] or [Location] in the
  output. Write natural English.
- Each bullet should be self-contained and actionable.
- 5 to 7 bullets total."""


CATEGORY_SKILLS_PROMPT = """You are an expert agent-memory summarizer.

You will be given a list of `planning_pattern` strings from successful trajectories of one specific task category: **{category_label}**. Each is a short action chain template.

Your job: produce 5-6 **Category-Specific Skills** that capture the
distinctive techniques for this task type. Each skill is a polished
English bullet plus a one-sentence "Apply when" condition.

Output format (strict JSON list, no preamble, no markdown fences):
[
  {{
    "name": "<Title Case Short Name, 2-4 words>",
    "description": "<one or two sentences ending with period; describes the skill in clear English specific to this task category>",
    "apply_when": "<one sentence ending with period; describes when to invoke this skill>"
  }},
  ...
]

Constraints:
- DO NOT use bracketed placeholders. Write natural English.
- Skills should be specific to the {category_label} task type, not generic.
- 5 to 6 bullets total."""


MISTAKES_PROMPT = """You are an expert agent-memory summarizer.

You will be given a list of `mistakes_to_avoid` items extracted from failed agent trajectories. Each item has a `trigger_condition` and a `bad_action`.

Your job: produce 5 **Mistakes to Avoid** entries that capture the most
common, generalizable failure modes across these failed trajectories.
Dedupe similar mistakes and polish into clear English.

Output format (strict JSON list, no preamble, no markdown fences):
[
  {
    "dont": "<one sentence describing the bad behavior, e.g. 'Agent repeatedly revisits ...'>",
    "instead": "<one sentence describing the correction, e.g. 'Maintain an exploration map ...'>"
  },
  ...
]

Constraints:
- DO NOT use bracketed placeholders. Write natural English.
- 5 entries exactly."""


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


def classify_alfworld_memory(mem: dict) -> str | None:
    """Classify a memory by inspecting its `contextual_description`."""
    desc = (mem.get("contextual_description") or "").lower()
    for cat in ALFWORLD_CATEGORIES:
        if cat in desc:
            return cat
    # Fall back to original_goal keywords
    goal = (mem.get("content", {}).get("task_meta", {}).get("original_goal") or "").lower()
    if "two" in goal:
        return "pick_two_obj_and_place"
    if "clean" in goal:
        return "pick_clean_then_place_in_recep"
    if "heat" in goal or "hot" in goal:
        return "pick_heat_then_place_in_recep"
    if "cool" in goal or "cold" in goal:
        return "pick_cool_then_place_in_recep"
    if "examine" in goal or "look at" in goal:
        return "look_at_obj_in_light"
    if "put" in goal or "place" in goal:
        return "pick_and_place"
    return None


def classify_webshop_memory(mem: dict) -> str:
    desc = (mem.get("contextual_description") or "").lower()
    apparel = ["shirt", "pant", "jeans", "dress", "shoe", "sock", "jacket", "coat", "hat", "apparel"]
    electronics = ["phone", "laptop", "tablet", "headphone", "earbud", "battery", "charger", "speaker", "camera", "electronics"]
    if any(kw in desc for kw in apparel):
        return "apparel"
    if any(kw in desc for kw in electronics):
        return "electronics"
    return "general"


def classify_search_memory(mem: dict) -> str:
    desc = (mem.get("contextual_description") or "").lower()
    if "multi-hop" in desc or "multi hop" in desc or "multihop" in desc:
        return "multi_hop"
    return "direct_retrieval"


def aggregate(memories: list[dict], env: str, client: OpenAI, model: str) -> dict:
    successes = [m for m in memories if m.get("tags", {}).get("outcome") == "Success"]
    failures = [m for m in memories if m.get("tags", {}).get("outcome") == "Failure"]

    # General principles
    print(f"Aggregating General Principles from {len(successes)} success memories...")
    all_patterns = [
        m.get("content", {}).get("strategic_guidelines", {}).get("planning_pattern", "")
        for m in successes
    ]
    all_patterns = [p for p in all_patterns if p]
    general = call_llm(
        client, model, GENERAL_PRINCIPLES_PROMPT, {"planning_patterns": all_patterns}
    )

    # Category skills
    if env == "alfworld":
        classifier = classify_alfworld_memory
        categories = ALFWORLD_CATEGORIES
    elif env == "webshop":
        classifier = classify_webshop_memory
        categories = WEBSHOP_CATEGORIES
    elif env == "search":
        classifier = classify_search_memory
        categories = SEARCH_CATEGORIES
    else:
        raise ValueError(f"Unknown env: {env}")

    by_cat: dict[str, list[str]] = {c: [] for c in categories}
    for m in successes:
        cat = classifier(m)
        if cat is None or cat not in by_cat:
            continue
        pat = m.get("content", {}).get("strategic_guidelines", {}).get("planning_pattern", "")
        if pat:
            by_cat[cat].append(pat)

    category_skills = {}
    for cat in categories:
        patterns = by_cat[cat]
        if not patterns:
            print(f"  [{cat}] no memories, skipping")
            category_skills[cat] = []
            continue
        if env == "alfworld":
            label = ALFWORLD_CATEGORY_DISPLAY.get(cat, cat)
        elif env == "search":
            label = SEARCH_CATEGORY_DISPLAY.get(cat, cat)
        else:
            label = cat.title()
        print(f"  [{cat}] aggregating from {len(patterns)} memories...")
        prompt = CATEGORY_SKILLS_PROMPT.format(category_label=label)
        category_skills[cat] = call_llm(client, model, prompt, {"planning_patterns": patterns})

    # Mistakes to avoid
    print(f"Aggregating Mistakes from {len(failures)} failure memories...")
    all_mistakes = []
    for m in failures:
        for mis in m.get("content", {}).get("strategic_guidelines", {}).get("mistakes_to_avoid", []) or []:
            all_mistakes.append(mis)
    if all_mistakes:
        mistakes = call_llm(client, model, MISTAKES_PROMPT, {"mistakes": all_mistakes})
    else:
        mistakes = []

    return {
        "general_principles": general,
        "category_skills": category_skills,
        "mistakes_to_avoid": mistakes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="generated_memories_*.json from stage 3")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--env", choices=["alfworld", "webshop", "search"], required=True)
    parser.add_argument("--model", default="gpt-4o")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY in the environment.")
    client = OpenAI(api_key=api_key)

    with open(args.input_file, "r", encoding="utf-8") as f:
        memories = json.load(f)

    skill_bank = aggregate(memories, args.env, client, args.model)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(skill_bank, f, indent=2, ensure_ascii=False)

    print(f"\nWrote skill bank to {args.output_file}")
    print(f"  general_principles: {len(skill_bank['general_principles'])} items")
    for cat, items in skill_bank["category_skills"].items():
        print(f"  category_skills[{cat}]: {len(items)} items")
    print(f"  mistakes_to_avoid: {len(skill_bank['mistakes_to_avoid'])} items")


if __name__ == "__main__":
    main()
