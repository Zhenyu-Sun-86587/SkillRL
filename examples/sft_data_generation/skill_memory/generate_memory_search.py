"""
Generate skill memories from preprocessed Search-agent trajectories.

Search-specific differences:
- The "trajectory" is a sequence of `<search>` / `<answer>` actions plus
  retrieved `<information>` snippets, not env actions + observations.
- Memories emphasize query patterns, multi-hop decomposition, and
  evidence-based answer composition.

Usage:
    export OPENAI_API_KEY=...
    python generate_memory_search.py \\
        --input_file processed_trajectories_search.json \\
        --output_file generated_memories_search.json \\
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
Your objective is to generate a **single-paragraph** "contextual_description" that describes the task and the resolution process.

**Requirements:**
* ONE plain-text paragraph (no bullets, no markdown).
* Include task type and what the agent did to solve it.
* Explicitly state success or failure and briefly why.
* Identify if it's a "Multi-hop reasoning" or "Direct retrieval" task.
* Keep it 1-3 sentences.

Return ONLY the paragraph string.
""",
    "refined_trajectory": """
You are an expert **Trajectory Refinement & Abstraction Engine** using a **"Backward Causal Chaining"** algorithm for Search/QA tasks.

**Phase 1:**
1. Find the step where the final `<answer>` was generated.
2. Trace backward: which `<search>` queries provided info used in the final answer?
3. Filter out queries that returned "No result" or whose info wasn't used.

**Phase 2:**
1. Generalize entities to placeholders (e.g., "Michael Strahan career" -> "Search for [Person] career history").
2. Summarize observations by type (e.g., "Found career dates and team list") rather than copying raw text.

**Output:** JSON list `refined_trajectory`.
* `step_index`, `action` (generalized query/reasoning), `critical_observation` (abstracted summary), `reasoning` (why necessary).

Output ONLY the JSON object.
""",
    "strategic_guidelines": """
You are an expert **Strategic Analyst** for a Search/QA Agent.

### **CASE 1: SUCCESS**
1. **`planning_pattern`:** logical chain (e.g., "Decompose Question -> Search Entity A -> Search Entity B -> Synthesize"). Use generic terms: `[Entity]`, `[Attribute]`, `[Time_Period]`.
2. **`mistakes_to_avoid`:** `[]`.

### **CASE 2: FAILURE**
1. **`planning_pattern`:** `null`.
2. **`mistakes_to_avoid`:** abstract `{trigger_condition, bad_action}` items. e.g., trigger "Ambiguous entity name", bad_action "Repeated same query".

Output ONLY the JSON object with keys `planning_pattern` and `mistakes_to_avoid`.
""",
}


def trajectory_to_string(steps):
    if not steps:
        return "No trajectory steps found."
    lines = []
    last_step = steps[-1]
    memory_context = (last_step.get("memory_context") or "").strip()
    if memory_context:
        lines.append(f"Memory Context:\n{memory_context}")
    for idx, step in enumerate(steps):
        step_id = step.get("step_id", idx)
        action = step.get("action", "")
        lines.append(f"Action {step_id}: {action}")
    return "\n".join(lines).strip()


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

    def create(self, env, goal, outcome, raw_traj_str, data_source):
        ctx = (
            f"**Input Data:**\nEnvironment: {env}\nGoal: {goal}\n"
            f"Outcome: {outcome}\nData Source: {data_source}\n"
            f"Raw Trajectory:\n{raw_traj_str}"
        )

        desc = (self._run(PROMPTS["contextual_description"], ctx) or "").strip().strip('"')

        refined = None
        if outcome.lower() == "success":
            refined = extract_json(self._run(PROMPTS["refined_trajectory"], ctx))

        strat = extract_json(self._run(PROMPTS["strategic_guidelines"], ctx))

        return {
            "memory_id": f"mem_search_{uuid.uuid4().hex[:8]}",
            "contextual_description": desc,
            "tags": {"environment": env, "outcome": outcome, "data_source": data_source},
            "content": {
                "task_meta": {"original_goal": goal, "data_source": data_source},
                "refined_trajectory": refined,
                "strategic_guidelines": strat,
            },
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--env_name", default="SearchAgent")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY in the environment.")
    client = OpenAI(api_key=api_key)

    with open(args.input_file, "r", encoding="utf-8") as f:
        entries = json.load(f)

    gen = MemoryGenerator(client, args.model)
    memories = []

    for i, entry in enumerate(entries):
        env_id = entry.get("env_id", f"task_{i}")
        goal = entry.get("task", "")
        outcome = entry.get("outcome", "Failure")
        data_source = entry.get("data_source")
        trajs = entry.get("trajectories", [])
        if not trajs:
            print(f"Skipping {env_id}: no trajectories")
            continue

        try:
            mem = gen.create(
                args.env_name, goal, outcome, trajectory_to_string(trajs[0]), data_source
            )
            mem["origin_env_id"] = env_id
            memories.append(mem)
            print(f"[{env_id}] [{outcome}] OK")
        except Exception as e:
            print(f"[{env_id}] FAILED: {e}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(memories, f, indent=2, ensure_ascii=False)

    cost = gen.input_tokens * PRICE_INPUT + gen.output_tokens * PRICE_OUTPUT
    print(f"\nSaved {len(memories)} memories to {args.output_file}")
    print(f"Estimated cost (gpt-4o pricing): ${cost:.4f}")


if __name__ == "__main__":
    main()
