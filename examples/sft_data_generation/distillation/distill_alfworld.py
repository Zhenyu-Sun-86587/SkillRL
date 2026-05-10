"""
Distill ALFWorld trajectories into ShareGPT SFT data with o3-generated reasoning.

Pipeline:
1. Load preprocessed (cleaned) successful trajectories.
2. For each trajectory:
    a. Retrieve relevant skill memories (by task category).
    b. Format the system prompt: task + 'Retrieved Relevant Experience' block.
    c. Build the per-step (human, gpt) ShareGPT conversation. Human turns
       follow the released format: an "[Observation N: ..., Action N: ...]"
       history (last `H` pairs), then "You are now at step K and your
       current observation is: ...", then the admissible-actions list and
       the standard reasoning instruction.
    d. **Append a synthetic terminal turn** so the agent can be supervised
       to emit `done`. The turn's observation reuses the env response to
       the winning action (which the env actually emitted before
       terminating); its admissible-actions list is sampled from the
       winning step's admissible with `done` injected at a random
       position.
    e. Call o3 ONCE per trajectory with the full step list and ask it to
       generate `<think>...</think>` reasoning for each turn. The action
       strings are taken verbatim from the trajectory (so action validity
       is guaranteed).
3. Write the distilled trajectories as a list of ShareGPT-formatted
   entries to `--output_file`.

Why we add the `done` step: ALFWorld terminates immediately at the
winning action and emits `done=True`. The original rollout never has a
step where `done` is in `admissible_commands`. Without the synthetic
step, the SFT'd agent would never learn to emit `done` and would keep
acting indefinitely after solving the task. This was the source of
confusion in https://github.com/aiming-lab/SkillRL/issues/27.

Trajectory schema assumption (from stage 2):
    parsed_steps[0]: {"step_id": "Step -1", "action": None,
                      "observation": <initial state>,
                      "admissible_actions": [...]}
    parsed_steps[k] (k>=1): {"step_id": "Step (k-1)", "action": <kth action>,
                             "observation": <state after kth action>,
                             "admissible_actions": [...]}

Conversation turn k (0-indexed) shows parsed_steps[k]'s obs+admissible
and outputs parsed_steps[k+1]'s action.

Usage:
    export OPENAI_API_KEY=...
    python distill_alfworld.py \\
        --input_file processed_trajectories_alfworld_cleaned.json \\
        --memory_file generated_memories_alfworld.json \\
        --output_file distilled_trajectories_alfworld.json \\
        --model o3 \\
        --max_history 5
"""
import argparse
import json
import os
import random
import re
import sys

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skill_retrieval import classify_alfworld_task, format_skills_block, load_skill_bank  # noqa: E402

ALFWORLD_SYSTEM_TEMPLATE = """You are an expert agent operating in the ALFRED Embodied Environment.
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


REASONING_PROMPT = """You will be given a successful ALFWorld trajectory: a list of steps, each with the agent's observation, the admissible actions available, and the action the agent took. The final step is a synthetic terminal step where the agent should emit the action `done` to mark task completion.

For EACH step, generate a single short `<think>...</think>` block that explains the agent's strategic reasoning for taking that step's action. The reasoning should:
- Refer to the task goal and current state.
- Cite relevant retrieved skills when applicable.
- Be 1-3 sentences. Avoid restating obvious observations.

Return a JSON list with exactly one entry per step, in order. Each entry has:
{
  "step_index": <int>,
  "think": "<the reasoning text without the <think> tags>"
}

Output ONLY the JSON. No preamble, no markdown fences."""


def sample_synthetic_done_admissible(winning_step: dict, rng: random.Random) -> list[str]:
    """Build the admissible-action list for the synthetic terminal turn.

    Sample up to 9 actions from the winning step's admissible (which is the
    state right after the winning action) and inject `done` at a random
    position. Inspection of the released SFT data suggests this matches the
    original construction.
    """
    prev = list(winning_step.get("admissible_actions") or [])
    keep = [a for a in prev if a != "done"]
    rng.shuffle(keep)
    keep = keep[:9]
    insert_at = rng.randrange(len(keep) + 1)
    keep.insert(insert_at, "done")
    return keep


def build_history_block(parsed_steps: list[dict], k: int, max_history: int) -> str:
    """Render the [Observation N: ..., Action N: ...] history block shown
    at turn k."""
    pairs = []
    start = max(0, k - max_history)
    for j in range(start, k):
        hist_obs = (parsed_steps[j].get("observation") or "").strip()
        hist_action = parsed_steps[j + 1].get("action", "")
        idx = j - start + 1
        pairs.append(f"[Observation {idx}: '{hist_obs}', Action {idx}: '{hist_action}']")
    return "\n".join(pairs)


def build_conversation(
    parsed_steps: list[dict],
    max_history: int,
    rng: random.Random,
):
    """Build (human, gpt) pairs.

    Yields one pair per real action plus one synthetic terminal pair for `done`.

    Returns: (convs, steps_summary) where:
        convs        — list of {"from", "value", "_step_index"} dicts
                       (with _step_index that the o3 reasoning step indexes into)
        steps_summary — what we send to o3 to ground its reasoning.
    """
    n_real = len(parsed_steps) - 1  # Number of real action turns
    convs = []
    summary = []

    for k in range(n_real):
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
                "is_synthetic_done": False,
            }
        )

    # Synthetic terminal turn
    winning_step = parsed_steps[-1]
    obs = (winning_step.get("observation") or "").strip()
    synthetic_admissible_list = sample_synthetic_done_admissible(winning_step, rng)
    admissible = ", ".join(synthetic_admissible_list)

    k = n_real
    history = build_history_block(parsed_steps, k, max_history)
    human_val = SUBSEQUENT_HUMAN_TEMPLATE.format(
        history=history, step_num=k + 1, obs=obs, admissible=admissible
    )
    convs.append({"from": "human", "value": human_val})
    convs.append(
        {
            "from": "gpt",
            "value": "<action>done</action>",
            "_step_index": k,
        }
    )
    summary.append(
        {
            "step_index": k,
            "observation": obs,
            "admissible_actions": synthetic_admissible_list,
            "action": "done",
            "is_synthetic_done": True,
        }
    )

    return convs, summary


def call_o3_for_reasoning(
    client: OpenAI,
    model: str,
    task: str,
    skills_block: str,
    steps_summary: list[dict],
) -> list[dict]:
    user_payload = {
        "task": task,
        "retrieved_skills": skills_block,
        "steps": steps_summary,
    }
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": REASONING_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    )
    text = resp.choices[0].message.content
    text = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def distill_one(
    client: OpenAI,
    model: str,
    task: str,
    parsed_steps: list[dict],
    skill_bank: dict,
    max_history: int,
    rng: random.Random,
) -> dict:
    category = classify_alfworld_task(task)
    skills_block = format_skills_block(skill_bank, env="alfworld", category=category)
    system_prompt = ALFWORLD_SYSTEM_TEMPLATE.format(task=task, skills_block=skills_block)

    convs, summary = build_conversation(parsed_steps, max_history, rng)
    reasonings = call_o3_for_reasoning(client, model, task, skills_block, summary)
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
    parser.add_argument("--input_file", required=True, help="Cleaned successful trajectories JSON")
    parser.add_argument(
        "--skill_bank_file",
        required=True,
        help="Aggregated skill bank from 03_skill_memory/aggregate_skills.py",
    )
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--model", default="o3")
    parser.add_argument("--max_history", type=int, default=5)
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling admissible actions in done step"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Distill at most N trajectories (for testing)"
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY in the environment.")
    client = OpenAI(api_key=api_key)

    rng = random.Random(args.seed)
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
                entry = distill_one(
                    client, args.model, task, steps, skill_bank, args.max_history, rng
                )
                distilled.append(entry)
                n_processed += 1
                print(f"[{env['env_id']}] OK ({entry['extra_info']['num_steps']} turns incl. done)")
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
