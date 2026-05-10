"""
Flatten distilled ShareGPT trajectories into per-step (instruction, output) pairs.

Input format (output of stage 3):
    {
      "system": "<system prompt with retrieved skills>",
      "conversations": [
        { "from": "human", "value": "<obs / admissible / prompt>" },
        { "from": "gpt",   "value": "<think>...</think><action>...</action>" },
        ...
      ],
      "extra_info": { ... }
    }

Output format (alpaca, matches HuggingFace SkillRL-SFT-Data schema):
    [
      { "instruction": "<system + Current Progress + history + obs>",
        "output": "<think>...</think>\\n<action>...</action>" },
      ...
    ]

The instruction layout for turn 0:
    {system_prompt}\\n\\n## Current Progress\\n{first_human_value}

For turn k > 0:
    {system_prompt}\\n\\n## Current Progress\\n\\nPrior to this step, you
    have already taken {k} step(s). Below are the most recent {H}
    observations and the corresponding actions you took:
    [Observation 1: ..., Action 1: ...]\\n[Observation 2: ...]\\n...
    \\nYou are now at step ... (+ rest of human turn)

This matches the released SFT data byte-for-byte (modulo the LLM-generated
skill bullets and `<think>` reasoning, which are non-deterministic).

Usage:
    python sharegpt_to_pairs.py \\
        --input_file distilled_trajectories.json \\
        --output_file alfworld_sft_data.json
"""
import argparse
import json
import os
import re

TEMPLATE_FIRST = """{system_prompt}

## Current Progress
{else_str}"""

TEMPLATE_SUBSEQUENT = """{system_prompt}

## Current Progress

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
{else_str}"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument(
        "--no_system_prepend",
        action="store_true",
        help="Disable system+Current Progress wrapping (use only the raw human turn). "
             "Rarely useful; provided for debugging.",
    )
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    aligned = []
    for entry in raw:
        system_content = entry.get("system", "")
        convs = entry.get("conversations", [])

        for i in range(0, len(convs), 2):
            if i + 1 >= len(convs):
                break
            user_val = convs[i]["value"]
            gpt_val = convs[i + 1]["value"]

            if args.no_system_prepend:
                full_prompt = user_val
            else:
                history_match = re.search(r"^(.*?)You are now at step", user_val, re.S)
                action_history = (
                    history_match.group(1).strip() if history_match else ""
                )
                if action_history:
                    else_str = user_val[len(action_history):].lstrip("\n")
                    history_count = action_history.count("[Observation")
                    full_prompt = TEMPLATE_SUBSEQUENT.format(
                        system_prompt=system_content,
                        step_count=i // 2,
                        history_length=history_count,
                        action_history=action_history,
                        else_str=else_str,
                    )
                else:
                    full_prompt = TEMPLATE_FIRST.format(
                        system_prompt=system_content, else_str=user_val
                    )

            aligned.append({"instruction": full_prompt, "output": gpt_val})

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(aligned, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(aligned)} examples to {args.output_file}")


if __name__ == "__main__":
    main()
