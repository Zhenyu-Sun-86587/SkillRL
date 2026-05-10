"""
Generate skill memories from preprocessed ALFWorld trajectories.

Each memory captures the strategic structure of a trajectory (success or
failure) so it can be retrieved at distillation time. Each entry has:

    - contextual_description: 1-line abstraction of the task + outcome
    - refined_trajectory:     causally-pruned action chain (success only)
    - strategic_guidelines:   planning_pattern + mistakes_to_avoid

We invoke the LLM (default gpt-4o, configurable) once per section per
trajectory. By default we only generate one memory per env (the first
trajectory in `trajectories`); pass --all_trajectories to generate one
per trajectory.

Usage:
    export OPENAI_API_KEY=...
    python generate_memory_alfworld.py \\
        --input_file processed_trajectories_alfworld_cleaned.json \\
        --output_file generated_memories_alfworld.json \\
        --model gpt-4o
"""
import argparse
import json
import os
import re
import uuid

from openai import OpenAI

PRICE_INPUT = 0.00125 / 1000
PRICE_OUTPUT = 0.0100 / 1000

PROMPTS = {
    "contextual_description": """
You are an expert **RAG Abstraction Engine** for an autonomous agent memory system.
Your objective is to generate a **"contextual_description"** that classifies the task type and summarizes the execution logic.

**Target Output Format:**
"AlfWorld task classified as '[Task Category]' to [Goal String]. [Outcome Description]."

**Rules:**
1. **Task Category (Strict):** exactly one of: `pick_and_place`, `pick_two_obj_and_place`, `look_at_obj_in_light`, `pick_heat_then_place_in_recep`, `pick_cool_then_place_in_recep`, `pick_clean_then_place_in_recep`.
2. **Goal String:** copy the exact 'Goal' string but remove all instance numbers/IDs.
3. **Outcome Description:**
    * IF Success: "Solved by [Sequential Action Summary]"
    * IF Failure: "Unsolved due to [Root Cause]"

Return ONLY the description string.
""",
    "refined_trajectory": """
You are an expert **Trajectory Refinement & Abstraction Engine** using a **"Backward Causal Chaining"** algorithm.
Your goal is to extract the minimal `refined_trajectory` from a raw log AND **generalize** the content into semantic placeholders.

**Phase 1: Refinement (Strictly Backward Logic)**
1. Start from the **Last Successful Step**. Call this `Current_Step`.
2. Recursively: identify the **Required Precondition** for `Current_Step`, scan backwards, find the nearest preceding step that produced it (`Preceding_Step`), prune everything between them, then set `Current_Step = Preceding_Step` and repeat.

**Phase 2: Abstraction**
1. Keep specific object names but **remove instance IDs** (e.g., "mug 1" -> "mug").
2. Replace specific values with `[*_Constraint]` placeholders.
3. Action/observation strings must use the abstracted entities.

**Output:** JSON list `refined_trajectory` (chronological).
* `step_index`: original index
* `action`: GENERALIZED action string
* `critical_observation`: GENERALIZED observation focusing on state changes
* `reasoning`: ONE highly-generalizable sentence on strategic intent. For exploration steps, emphasize **search for unknown** rather than "Navigate".

Output ONLY the JSON object.
""",
    "strategic_guidelines_alfworld": """
You are an expert **Strategic Analyst**.
Your goal is to extract high-level `strategic_guidelines` focusing on the execution skeleton and error avoidance.

### **CASE 1: SUCCESS**
1. **`planning_pattern`:** Logical chain using " -> " separators.
   - **NEVER** use specific names. Replace objects with `[Object_1]`, `[Object_2]`, search areas with `[Location]`, fixed destinations with `[Target_Location]`.
   - **Mandatory Search Precondition:** before interacting with any object, prefix with **"Search [Location] where possibly having [Object_X]"** -> **"[Action_Verb] [Object_X]"**. Use verbs like Acquire / Use / Heat / Cool / Clean / Place where they fit.
   - **Multi-Object:** if same location, combine searches; if different, distinguish.
   - **Navigation:** `"Navigate to [Target_Location]"` ONLY for fixed appliance/placement destinations after acquiring the object.
2. **`mistakes_to_avoid`:** `[]`.

### **CASE 2: FAILURE**
1. **`planning_pattern`:** `null`.
2. **`mistakes_to_avoid`:** abstract list of `{trigger_condition, bad_action}` using `[Target_Object]`, `[Container]`, `[Location]` etc. Never specific instance IDs.

Output ONLY the JSON object with keys `planning_pattern` and `mistakes_to_avoid`.
""",
}


def trajectory_to_string(steps):
    out = []
    for s in steps:
        out.append(
            f"{s.get('step_id', 'Step ?')} | Action: {s.get('action', 'None')} | "
            f"Reward: {s.get('reward', 0.0)} | Done: {s.get('done', False)}\n"
            f"Obs: {(s.get('observation') or '').strip()}\n"
        )
    return "\n".join(out)


def extract_json(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        for pat in (r"```json\s*(.*?)\s*```", r"(\{.*\}|\[.*\])"):
            m = re.search(pat, text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    continue
    return None


class MemoryGenerator:
    def __init__(self, client, model_name):
        self.client = client
        self.model = model_name
        self.input_tokens = 0
        self.output_tokens = 0

    def _run(self, system_prompt, user_content):
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0,
            )
            self.input_tokens += resp.usage.prompt_tokens
            self.output_tokens += resp.usage.completion_tokens
            return resp.choices[0].message.content
        except Exception as e:
            print(f"LLM error: {e}")
            return None

    def create(self, env, goal, outcome, raw_traj_str):
        ctx = (
            f"**Input Data:**\nEnvironment: {env}\nGoal: {goal}\n"
            f"Outcome: {outcome}\nRaw Trajectory:\n{raw_traj_str}"
        )

        description = (self._run(PROMPTS["contextual_description"], ctx) or "").strip().strip('"')

        refined = None
        if outcome.lower() == "success":
            refined = extract_json(self._run(PROMPTS["refined_trajectory"], ctx))

        strategic = extract_json(self._run(PROMPTS["strategic_guidelines_alfworld"], ctx))

        return {
            "memory_id": f"mem_{env.lower()}_{uuid.uuid4().hex[:8]}",
            "contextual_description": description,
            "tags": {"environment": env, "outcome": outcome},
            "content": {
                "task_meta": {"original_goal": goal},
                "refined_trajectory": refined,
                "strategic_guidelines": strategic,
            },
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--env_name", default="Alfworld")
    parser.add_argument(
        "--all_trajectories",
        action="store_true",
        help="Generate one memory per trajectory (default: only the first per env).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY in the environment.")
    client = OpenAI(api_key=api_key)

    with open(args.input_file, "r", encoding="utf-8") as f:
        entries = json.load(f)

    gen = MemoryGenerator(client, args.model)
    memories = []

    for entry in entries:
        env_id = entry.get("env_id", "Unknown")
        goal = entry.get("task", "")
        trajs = entry.get("trajectories", [])
        outcome = "Success" if entry.get("type", "") == "all_success" else "Failure"

        for idx, steps in enumerate(trajs):
            try:
                mem = gen.create(args.env_name, goal, outcome, trajectory_to_string(steps))
                mem["origin_env_id"] = env_id
                memories.append(mem)
                print(f"[{env_id} traj {idx}] [{outcome}] OK")
            except Exception as e:
                print(f"[{env_id} traj {idx}] FAILED: {e}")
            if not args.all_trajectories:
                break

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(memories, f, indent=2, ensure_ascii=False)

    cost = gen.input_tokens * PRICE_INPUT + gen.output_tokens * PRICE_OUTPUT
    print(f"\nSaved {len(memories)} memories to {args.output_file}")
    print(f"Estimated cost (gpt-4o pricing): ${cost:.4f}")


if __name__ == "__main__":
    main()
