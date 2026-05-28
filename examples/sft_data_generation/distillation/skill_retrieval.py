"""
Format the 'Retrieved Relevant Experience' system-prompt block from a
pre-aggregated skill bank.

The skill bank (produced by `skill_memory/aggregate_skills.py`) follows the
same schema as `skill_generation/{alfworld,webshop,search}.py`:

    {
      "general_skills":       [{"skill_id", "title", "principle", "when_to_apply"}, ...],
      "task_specific_skills": {"<cat>": [...], ...},   # alfworld + webshop
      "query_type_skills":    {"<cat>": [...], ...},   # search only
      "common_mistakes":      [{"mistake_id", "description", "why_it_happens",
                                "how_to_avoid"}, ...]
    }

Distillation looks up the right category here; no per-trajectory retrieval
is performed.
"""
import json

# ---------------------------------------------------------------------------
# ALFWorld
# ---------------------------------------------------------------------------

# Order matters: longer/more-specific patterns first.
ALFWORLD_TASK_KEYWORDS = [
    ("look_at_obj_in_light", ["look at", "examine"]),
    ("clean", ["clean"]),
    ("heat", ["heat", "hot"]),
    ("cool", ["cool", "cold"]),
    # `pick_and_place` is the catch-all for "put"/"place" goals, including
    # the two-object variant.
    ("pick_and_place", ["put", "place"]),
]

ALFWORLD_CATEGORY_DISPLAY = {
    "pick_and_place": "Pick And Place",
    "look_at_obj_in_light": "Look At Object In Light",
    "clean": "Clean",
    "heat": "Heat",
    "cool": "Cool",
    "examine": "Examine",
}


# ---------------------------------------------------------------------------
# WebShop — keyword table matches `skill_generation/webshop.py`
# ---------------------------------------------------------------------------

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

WEBSHOP_CATEGORY_DISPLAY = {
    "apparel": "Apparel",
    "footwear": "Footwear",
    "home_decor": "Home Decor",
    "electronics": "Electronics",
    "accessories": "Accessories",
    "beauty_health": "Beauty Health",
    "other": "Other",
}


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

SEARCH_CATEGORY_DISPLAY = {
    "direct_retrieval": "Direct Retrieval",
    "multi_hop_reasoning": "Multi-Hop Reasoning",
    "entity_attribute_lookup": "Entity Attribute Lookup",
    "comparison": "Comparison",
}


# Which top-level key holds the per-category skills for each env.
ENV_CATEGORY_KEY = {
    "alfworld": "task_specific_skills",
    "webshop": "task_specific_skills",
    "search": "query_type_skills",
}


# ---------------------------------------------------------------------------
# Classifiers — take a task / question string and return a category key.
# ---------------------------------------------------------------------------

def classify_alfworld_task(task: str) -> str:
    task_l = task.lower()
    for category, keywords in ALFWORLD_TASK_KEYWORDS:
        if any(kw in task_l for kw in keywords):
            return category
    return "pick_and_place"


def classify_webshop_category(task: str) -> str:
    task_l = task.lower()
    for category, keywords in WEBSHOP_KEYWORD_TABLE:
        if any(kw in task_l for kw in keywords):
            return category
    return "other"


def classify_search_question(question: str) -> str:
    """Classify a question into one of `skill_generation/search.py`'s 4 query types.

    `data_source` would be more reliable but is not available at distillation
    time, so fall back to surface cues in the question text.
    """
    q = question.lower()
    comparison_kw = [
        "both", "are the", "which of", "same", "common", "more than", "less than",
        "older", "younger", "taller", "shorter",
    ]
    multi_hop_kw = [
        " who is the ", " where was ", " when was the spouse",
        " father of", " mother of", "directed by", "starring ",
    ]
    if any(kw in q for kw in comparison_kw):
        return "comparison"
    if any(kw in q for kw in multi_hop_kw):
        return "multi_hop_reasoning"
    return "direct_retrieval"


# ---------------------------------------------------------------------------
# Skill bank loading + rendering
# ---------------------------------------------------------------------------

def load_skill_bank(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_memories(path: str) -> list[dict]:
    """Backward-compat for older memory-based callers."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_skills_block(
    skill_bank: dict,
    *,
    env: str,
    category: str,
) -> str:
    """Render the 'Retrieved Relevant Experience' block from the bank.

    `category` selects which per-category skill list to include — use a key
    from ALFWORLD_CATEGORY_DISPLAY / WEBSHOP_CATEGORY_DISPLAY /
    SEARCH_CATEGORY_DISPLAY depending on `env`.
    """
    out = ["## Retrieved Relevant Experience", "", "### General Principles"]
    for p in skill_bank.get("general_skills", []):
        title = p.get("title", "")
        principle = (p.get("principle") or "").rstrip(".")
        out.append(f"- **{title}**: {principle}.")

    category_key = ENV_CATEGORY_KEY.get(env, "task_specific_skills")
    cat_skills = skill_bank.get(category_key, {}).get(category, [])
    if cat_skills:
        if env == "alfworld":
            label = ALFWORLD_CATEGORY_DISPLAY.get(category, category.replace("_", " ").title())
        elif env == "search":
            label = SEARCH_CATEGORY_DISPLAY.get(category, category.replace("_", " ").title())
        else:
            label = WEBSHOP_CATEGORY_DISPLAY.get(category, category.replace("_", " ").title())
        out.append("")
        out.append(f"### {label} Skills")
        for s in cat_skills:
            title = s.get("title", "")
            principle = (s.get("principle") or "").rstrip(".")
            when = (s.get("when_to_apply") or "").rstrip(".")
            out.append(f"- **{title}**: {principle}.")
            if when:
                out.append(f"  _Apply when: {when}._")

    mistakes = skill_bank.get("common_mistakes", [])
    if mistakes:
        out.append("")
        out.append("### Mistakes to Avoid")
        for mis in mistakes:
            desc = (mis.get("description") or "").rstrip(".")
            fix = (mis.get("how_to_avoid") or "").rstrip(".")
            if not desc:
                continue
            out.append(f"- **Don't**: {desc}.")
            if fix:
                out.append(f"  **Instead**: {fix}.")

    return "\n".join(out)


__all__ = [
    "classify_alfworld_task",
    "classify_webshop_category",
    "classify_search_question",
    "format_skills_block",
    "load_skill_bank",
    "load_memories",
]
