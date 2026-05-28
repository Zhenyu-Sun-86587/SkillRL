"""
Distill WebShop trajectories into ShareGPT SFT data with o3-generated reasoning.

Differs from ALFWorld in three ways:
1. **No synthetic terminal step**: WebShop's winning action `click[buy
   now]` already produces a `Thank you for shopping...` observation and
   terminates the env naturally. The model just needs to learn to emit
   `click[buy now]` at the right time.
2. **Different system prompt**: WebShop-specific phrasing ("operating in
   the WebShop e-commerce environment").
3. **Skill retrieval is by item category** (e.g., "Apparel",
   "Electronics") rather than ALFWorld task types.

Trajectory schema assumption (from stage 2):
    parsed_steps[0]: {"step_id": "Step -1", "action": None,
                      "observation": <initial obs>, "admissible_actions": [...]}
    parsed_steps[k] (k>=1): {"action": <kth action>,
                             "observation": <state after kth action>, ...}

Conversation turn k shows parsed_steps[k]'s obs+admissible and outputs
parsed_steps[k+1]'s action.

Usage:
    export OPENAI_API_KEY=...
    python distill_webshop.py \\
        --input_file processed_trajectories_webshop_cleaned.json \\
        --memory_file generated_memories_webshop.json \\
        --output_file distilled_trajectories_webshop.json \\
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
    classify_webshop_category,
    format_skills_block,
    load_skill_bank,
)

WEBSHOP_SYSTEM_TEMPLATE = """You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task}

{skills_block}"""

INITIAL_HUMAN_TEMPLATE = """Your current observation is: {obs}
Your admissible actions of the current situation are: [{admissible}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""

SUBSEQUENT_HUMAN_TEMPLATE = """{history}
You are now at step {step_num} and your current observation is: {obs}
Your admissible actions of the current situation are: [{admissible}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""


REASONING_PROMPT = """You will be given a successful WebShop trajectory: a list of steps, each with the agent's observation, the admissible actions available, and the action the agent took. The trajectory ends with `click[buy now]` after the agent has selected matching options.

For EACH step, generate a single short `<think>...</think>` block explaining the agent's strategic reasoning. Cite key product attributes and constraints from the task. Be 1-3 sentences.

Return a JSON list with one entry per step, in order:
{
  "step_index": <int>,
  "think": "<reasoning text without the <think> tags>"
}

Output ONLY the JSON. No preamble, no markdown fences."""


def build_history_block(parsed_steps, k, max_history):
    pairs = []
    start = max(0, k - max_history)
    for j in range(start, k):
        hist_obs = (parsed_steps[j].get("observation") or "").strip()
        hist_action = parsed_steps[j + 1].get("action", "")
        idx = j - start + 1
        pairs.append(f"[Observation {idx}: '{hist_obs}', Action {idx}: '{hist_action}']")
    return "\n".join(pairs)


def build_conversation(parsed_steps, max_history):
    n_turns = len(parsed_steps) - 1
    convs = []
    summary = []
    for k in range(n_turns):
        obs_step = parsed_steps[k]
        action_step = parsed_steps[k + 1]
        obs = (obs_step.get("observation") or "").strip()
        admissible_list = obs_step.get("admissible_actions") or []
        admissible = ", ".join(admissible_list)

        if k == 0:
            human_val = INITIAL_HUMAN_TEMPLATE.format(obs=obs, admissible=admissible)
        else:
            history = build_history_block(parsed_steps, k, max_history)
            human_val = SUBSEQUENT_HUMAN_TEMPLATE.format(
                history=history, step_num=k + 1, obs=obs, admissible=admissible
            )

        convs.append({"from": "human", "value": human_val})
        convs.append(
            {
                "from": "gpt",
                "value": f"<action>{action_step['action']}</action>",
                "_step_index": k,
            }
        )
        summary.append(
            {
                "step_index": k,
                "observation": obs,
                "admissible_actions": admissible_list,
                "action": action_step["action"],
            }
        )
    return convs, summary


def call_o3(client, model, task, skills_block, steps_summary):
    payload = {"task": task, "retrieved_skills": skills_block, "steps": steps_summary}
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


def distill_one(client, model, task, parsed_steps, skill_bank, max_history):
    category = classify_webshop_category(task)
    skills_block = format_skills_block(skill_bank, env="webshop", category=category)
    system_prompt = WEBSHOP_SYSTEM_TEMPLATE.format(task=task, skills_block=skills_block)

    convs, summary = build_conversation(parsed_steps, max_history)
    reasonings = call_o3(client, model, task, skills_block, summary)
    reason_by_idx = {int(r["step_index"]): r["think"] for r in reasonings}

    for c in convs:
        if c["from"] == "gpt":
            idx = c.pop("_step_index")
            think = reason_by_idx.get(idx, "")
            c["value"] = f"<think>{think}</think>\n{c['value']}"

    return {
        "system": system_prompt,
        "conversations": convs,
        "extra_info": {
            "task": task,
            "task_type": category,
            "num_steps": len(summary),
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
    parser.add_argument("--max_history", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY in the environment.")
    client = OpenAI(api_key=api_key)

    skill_bank = load_skill_bank(args.skill_bank_file)
    with open(args.input_file, "r", encoding="utf-8") as f:
        envs = json.load(f)

    distilled = []
    n_processed = 0
    for env in envs:
        if env.get("type") != "all_success":
            continue
        task = env.get("task", "")
        for steps in env.get("trajectories", []):
            if len(steps) < 2:
                continue
            try:
                distilled.append(
                    distill_one(client, args.model, task, steps, skill_bank, args.max_history)
                )
                n_processed += 1
                print(f"[{env['env_id']}] OK ({len(steps)-1} turns)")
            except Exception as e:
                print(f"[{env['env_id']}] FAILED: {e}")
            if args.limit and n_processed >= args.limit:
                break
        if args.limit and n_processed >= args.limit:
            break

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(distilled, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(distilled)} distilled trajectories to {args.output_file}")


if __name__ == "__main__":
    main()
