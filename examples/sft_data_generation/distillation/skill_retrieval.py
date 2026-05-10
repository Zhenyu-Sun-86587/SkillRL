"""
Format the 'Retrieved Relevant Experience' system-prompt block from a
pre-aggregated skill bank.

The skill bank (produced by `03_skill_memory/aggregate_skills.py`) is a
single JSON file with three sections:

    {
      "general_principles": [{"name", "description"}, ...],
      "category_skills":    {"<cat>": [{"name", "description", "apply_when"}, ...]},
      "mistakes_to_avoid":  [{"dont", "instead"}, ...]
    }

The released SFT data shows that General Principles and Mistakes to Avoid
are STATIC across all tasks of an env, while Category Skills varies by
task category. Distillation just looks up the right category here; no
per-trajectory retrieval is performed.
"""
import json

ALFWORLD_TASK_KEYWORDS = {
    "pick_two_obj_and_place": ["two"],
    "pick_clean_then_place_in_recep": ["clean"],
    "pick_heat_then_place_in_recep": ["heat", "hot"],
    "pick_cool_then_place_in_recep": ["cool", "cold"],
    "look_at_obj_in_light": ["look at", "examine"],
    "pick_and_place": ["put", "place"],
}

ALFWORLD_CATEGORY_DISPLAY = {
    "pick_and_place": "Pick And Place",
    "pick_two_obj_and_place": "Pick And Place",
    "look_at_obj_in_light": "Examine",
    "pick_heat_then_place_in_recep": "Heat",
    "pick_cool_then_place_in_recep": "Cool",
    "pick_clean_then_place_in_recep": "Clean",
}

WEBSHOP_CATEGORY_DISPLAY = {
    "general": "General",
    "apparel": "Apparel",
    "electronics": "Electronics",
}

SEARCH_CATEGORY_DISPLAY = {
    "direct_retrieval": "Direct Retrieval",
    "multi_hop": "Multi-Hop Reasoning",
}


def classify_search_question(question: str) -> str:
    """Heuristic: multi-hop if the question asks for a chained relationship."""
    q = question.lower()
    multi_hop_kw = [" who is the ", " where was ", " when was the spouse", " father of", " mother of", "directed by", "starring "]
    if any(kw in q for kw in multi_hop_kw):
        return "multi_hop"
    return "direct_retrieval"


def classify_alfworld_task(task: str) -> str:
    task_l = task.lower()
    for category, keywords in ALFWORLD_TASK_KEYWORDS.items():
        if any(kw in task_l for kw in keywords):
            return category
    return "pick_and_place"


def load_skill_bank(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_memories(path: str) -> list[dict]:
    """Backward-compat for the older memory-based callers."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_skills_block(
    skill_bank: dict,
    *,
    env: str,
    category: str,
) -> str:
    """Render the 'Retrieved Relevant Experience' block from the bank.

    `category` selects which Category Skills section to include. For
    `env="alfworld"` use one of the ALFWORLD_TASK_KEYWORDS keys; for
    `env="webshop"` use 'general' / 'apparel' / 'electronics'.
    """
    out = ["## Retrieved Relevant Experience", "", "### General Principles"]
    for p in skill_bank.get("general_principles", []):
        name = p.get("name", "")
        desc = p.get("description", "").rstrip(".")
        out.append(f"- **{name}**: {desc}.")

    cat_skills = skill_bank.get("category_skills", {}).get(category, [])
    if cat_skills:
        if env == "alfworld":
            label = ALFWORLD_CATEGORY_DISPLAY.get(category, category.replace("_", " ").title())
        elif env == "search":
            label = SEARCH_CATEGORY_DISPLAY.get(category, category.replace("_", " ").title())
        else:
            label = WEBSHOP_CATEGORY_DISPLAY.get(category, category.title())
        out.append("")
        out.append(f"### {label} Skills")
        for s in cat_skills:
            name = s.get("name", "")
            desc = s.get("description", "").rstrip(".")
            apply_when = (s.get("apply_when") or "").rstrip(".")
            out.append(f"- **{name}**: {desc}.")
            if apply_when:
                out.append(f"  _Apply when: {apply_when}._")

    mistakes = skill_bank.get("mistakes_to_avoid", [])
    if mistakes:
        out.append("")
        out.append("### Mistakes to Avoid")
        for mis in mistakes:
            dont = mis.get("dont", "").rstrip(".")
            instead = mis.get("instead", "").rstrip(".")
            out.append(f"- **Don't**: {dont}.")
            if instead:
                out.append(f"  **Instead**: {instead}.")

    return "\n".join(out)


__all__ = [
    "classify_alfworld_task",
    "classify_search_question",
    "format_skills_block",
    "load_skill_bank",
    "load_memories",
]
