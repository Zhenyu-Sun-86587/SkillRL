"""
Generate skill memories from preprocessed WebShop trajectories.

Same overall structure as ALFWorld but with WebShop-specific prompts that
emphasize product categories, attribute matching, and price/size/color
constraints.

Usage:
    export OPENAI_API_KEY=...
    python generate_memory_webshop.py \\
        --input_file processed_trajectories_webshop_cleaned.json \\
        --output_file generated_memories_webshop.json \\
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
"WebShop task to [Action] a [Item Category] with [Constraint Types]. [Outcome Description]."

**Rules:**
1. **Item Category:** map specific items to broad domains (e.g., "Electronics", "Apparel").
2. **Constraint Types:** describe categories only (e.g., "Price and Attribute constraints"). NO specific values.
3. **Outcome Description:**
    * IF Success: "Solved by [Sequential Action Summary]" (e.g., "Solved by searching full terms, selecting options, and buying.")
    * IF Failure: "Unsolved due to [Root Cause]"

Return ONLY the description string.
""",
    "refined_trajectory": """
You are an expert **Trajectory Refinement & Abstraction Engine** using a **"Backward Causal Chaining"** algorithm.

**Phase 1:** Identify the Last Successful Step (typically `click[buy now]`). Recursively trace back to its required preconditions; prune intermediate noise.

**Phase 2 (Abstraction for WebShop):**
1. Replace specific product titles or IDs with the **Broad Item Category** (e.g., "Samsung Galaxy S21" -> "Smartphone").
2. **NEVER** output specific values (prices, sizes, colors). Replace with `[Price_Constraint]`, `[Size_Constraint]`, `[Color_Constraint]`, etc.
3. Action strings: e.g., `click[[Size_Constraint]]`.
4. Observation summaries describe the state change using abstracted terms.

**Output:** JSON list `refined_trajectory` (chronological).
* `step_index`, `action` (generalized), `critical_observation` (generalized), `reasoning` (single generalizable sentence).

Output ONLY the JSON object.
""",
    "strategic_guidelines_webshop": """
You are an expert **Strategic Analyst** for a WebShop agent.

### **CASE 1: SUCCESS**
1. **`planning_pattern`:** `ActionType -> ActionType -> ActionType` skeleton (e.g., "Search -> Filter -> Select -> Verify -> Buy").
2. **`mistakes_to_avoid`:** `[]`.

### **CASE 2: FAILURE**
1. **`planning_pattern`:** `null`.
2. **`mistakes_to_avoid`:** abstract `{trigger_condition, bad_action}` items.

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
    def __init__(self, client, model):
        self.client = client
        self.model = model
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

        desc = (self._run(PROMPTS["contextual_description"], ctx) or "").strip().strip('"')

        refined = None
        if outcome.lower() == "success":
            refined = extract_json(self._run(PROMPTS["refined_trajectory"], ctx))

        strat = extract_json(self._run(PROMPTS["strategic_guidelines_webshop"], ctx))

        return {
            "memory_id": f"mem_webshop_{uuid.uuid4().hex[:8]}",
            "contextual_description": desc,
            "tags": {"environment": env, "outcome": outcome},
            "content": {
                "task_meta": {"original_goal": goal},
                "refined_trajectory": refined,
                "strategic_guidelines": strat,
            },
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--env_name", default="WebShop")
    parser.add_argument("--all_trajectories", action="store_true")
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
