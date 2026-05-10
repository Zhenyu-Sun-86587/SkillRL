"""
Distill Search-agent trajectories into ShareGPT SFT data with o3 reasoning.

Search-agent specifics:
- Actions are `<search>...</search>` or `<answer>...</answer>` tags rather
  than env actions with admissible-action lists.
- Each step's "observation" is the retrieved `<information>...</information>`
  block from the previous search; the trajectory ends when the agent emits
  `<answer>`.
- No synthetic terminal step is required (the trajectory naturally ends
  with `<answer>`).
- The system prompt is a single-line "You are a helpful and harmless
  assistant." plus a per-turn human prompt that contains the question and
  retrieved skills.

Usage:
    export OPENAI_API_KEY=...
    python distill_search.py \\
        --input_file processed_trajectories_search.json \\
        --memory_file generated_memories_search.json \\
        --output_file distilled_trajectories_search.json \\
        --model o3
"""
import argparse
import json
import os
import re
import sys

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skill_retrieval import (  # noqa: E402
    classify_search_question,
    format_skills_block,
    load_skill_bank,
)

SEARCH_SYSTEM = "You are a helpful and harmless assistant."

# A search-agent prompt is a self-contained user turn rather than a
# system+user pair, mirroring the released SFT data.
SEARCH_HUMAN_TEMPLATE = """You are an expert agent tasked with answering the given question step-by-step.
Your question: {question}

{skills_block}

{history_section}You should first reason step-by-step inside <think> </think> tags. Then either issue another search via <search>...</search> or commit to a final answer via <answer>...</answer>."""

REASONING_PROMPT = """You will be given a successful Search trajectory: a sequence of `<search>` queries that retrieved evidence, followed by a final `<answer>`.

For EACH step generate a single short `<think>...</think>` block (1-3 sentences) explaining the agent's strategic reasoning: what sub-question it's tackling, what evidence it has gathered, and why it chose this action.

Return a JSON list with one entry per step, in order:
{
  "step_index": <int>,
  "think": "<reasoning text without the <think> tags>"
}

Output ONLY the JSON. No preamble, no markdown fences."""


def build_human_turn(question: str, skills_block: str, prior_steps: list[dict]) -> str:
    if not prior_steps:
        history_section = ""
    else:
        history_lines = []
        for s in prior_steps:
            action = s.get("action", "")
            history_lines.append(f"Step {s.get('step_id', '?')}: {action}")
        history_section = "History:\n" + "\n".join(history_lines) + "\n\n"

    return SEARCH_HUMAN_TEMPLATE.format(
        question=question, skills_block=skills_block, history_section=history_section
    )


def call_o3(client, model, question, skills_block, steps_summary):
    payload = {
        "question": question,
        "retrieved_skills": skills_block,
        "steps": steps_summary,
    }
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": REASONING_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    text = resp.choices[0].message.content
    text = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        return json.loads(m.group(0))


def steps_to_summary(steps):
    return [
        {
            "step_index": i,
            "memory_context": s.get("memory_context", ""),
            "action": s["action"],
        }
        for i, s in enumerate(steps)
    ]


def distill_one(client, model, question, steps, skill_bank):
    category = classify_search_question(question)
    skills_block = format_skills_block(skill_bank, env="search", category=category)

    convs = []
    for i, step in enumerate(steps):
        human_val = build_human_turn(question, skills_block, steps[:i])
        action = step.get("action", "")
        convs.append({"from": "human", "value": human_val})
        convs.append({"from": "gpt", "value": action, "_step_index": i})

    reasonings = call_o3(client, model, question, skills_block, steps_to_summary(steps))
    reason_by_idx = {int(r["step_index"]): r["think"] for r in reasonings}

    for c in convs:
        if c["from"] == "gpt":
            idx = c.pop("_step_index")
            think = reason_by_idx.get(idx, "")
            c["value"] = f"<think>{think}</think>\n{c['value']}"

    return {
        "system": SEARCH_SYSTEM,
        "conversations": convs,
        "extra_info": {
            "task": question,
            "num_steps": len(steps),
            "source": f"{model}_distillation",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument(
        "--skill_bank_file",
        required=True,
        help="Aggregated skill bank from 03_skill_memory/aggregate_skills.py",
    )
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--model", default="o3")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY in the environment.")
    client = OpenAI(api_key=api_key)

    skill_bank = load_skill_bank(args.skill_bank_file)
    with open(args.input_file, "r", encoding="utf-8") as f:
        entries = json.load(f)

    distilled = []
    n_processed = 0
    for entry in entries:
        if entry.get("outcome") != "Success":
            continue
        question = entry.get("task", "")
        trajs = entry.get("trajectories", [])
        if not trajs or not trajs[0]:
            continue
        try:
            distilled.append(distill_one(client, args.model, question, trajs[0], skill_bank))
            n_processed += 1
            print(f"[{entry.get('env_id')}] OK ({len(trajs[0])} steps)")
        except Exception as e:
            print(f"[{entry.get('env_id')}] FAILED: {e}")
        if args.limit and n_processed >= args.limit:
            break

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(distilled, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(distilled)} distilled trajectories to {args.output_file}")


if __name__ == "__main__":
    main()
