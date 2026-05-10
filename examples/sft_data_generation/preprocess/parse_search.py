"""
Parse raw Search-agent rollout trajectories (txt format) into structured JSON.

Search-agent rollouts use a different on-disk format than ALFWorld/WebShop:
- The trajectory is encoded as a History block followed by 'Now it's your
  turn' and a final action.
- Each step is `Step <i>: <search/answer tag>` plus an optional
  `<information>...</information>` block (the retrieved document).
- 'Reward' marks the final reward (1.0 for success, else failure).

Output schema (one record per .txt file):
    {
      "env_id": "<envFolder>_<filename>",
      "task": "<question>",
      "data_source": "nq | popqa | 2wikimultihopqa | triviaqa | hotpotqa | musique | bamboogle",
      "type": "all_success" | "all_fail",
      "outcome": "Success" | "Failure",
      "reward": float,
      "trajectories": [[step_dict, ...]]
    }

Each step contains `step_id`, `action` (the <search>/<answer> tag), and
`memory_context` — a string in the same format the agent would see at
inference, replaying prior steps' actions and information snippets.

Filtering:
- Per data_source, keep ALL successes and only the first
  `failure_ratio * n_success` failures (sorted by env_id), to roughly
  balance the dataset.

Usage:
    python parse_search.py \\
        --input_dir /path/to/trajectories \\
        --output_file processed_trajectories_search.json \\
        --failure_ratio 0.5
"""
import argparse
import glob
import json
import os
import re

PATTERN_QUESTION = re.compile(r"Your question:\s*(.*?)(?:\n|$)", re.DOTALL)
PATTERN_REWARD = re.compile(r"Rewards?:\s*([\d\.]+)")
PATTERN_HISTORY_STEP = re.compile(
    r"Step (\d+):\s*"
    r"(<(?:search|answer)>.*?</(?:search|answer)>)"
    r"\s*"
    r"(?:<information>(.*?)</information>)?",
    re.DOTALL,
)
PATTERN_FINAL_ACTION = re.compile(
    r"Action \d+:\s*(<(?:search|answer)>.*?</(?:search|answer)>)", re.DOTALL
)


def infer_data_source(file_path: str) -> str:
    """Infer the QA dataset from the file index. Adjust the ranges to match
    your rollout numbering scheme."""
    base_name = os.path.basename(file_path)
    try:
        idx = int(os.path.splitext(base_name)[0])
    except ValueError:
        return "unknown"
    ranges = [
        (0, 80, "nq"),
        (81, 160, "popqa"),
        (161, 240, "2wikimultihopqa"),
        (241, 320, "triviaqa"),
        (321, 400, "hotpotqa"),
        (401, 480, "musique"),
    ]
    for lo, hi, src in ranges:
        if lo <= idx <= hi:
            return src
    return "bamboogle"


def parse_search_trajectory_file(file_path: str):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"[Error] reading {file_path}: {e}")
        return None

    q_match = PATTERN_QUESTION.search(content)
    task_desc = q_match.group(1).strip() if q_match else "Unknown Task"

    r_match = PATTERN_REWARD.search(content)
    reward = float(r_match.group(1)) if r_match else 0.0
    is_success = reward == 1.0

    steps = []
    accumulated_history_str = ""

    history_matches = list(
        re.finditer(r"History:\s*\n(.*?)\nNow it's your turn", content, re.DOTALL)
    )
    if history_matches:
        last_history_block = history_matches[-1].group(1)
        for m in PATTERN_HISTORY_STEP.finditer(last_history_block):
            step_id = int(m.group(1))
            action_str = m.group(2).strip()
            raw_obs_str = m.group(3).strip() if m.group(3) else ""

            steps.append(
                {
                    "step_id": step_id,
                    "action": action_str,
                    "memory_context": accumulated_history_str,
                    "reward": 0.0,
                    "done": False,
                }
            )

            obs_tag = f" <information>{raw_obs_str}</information>" if raw_obs_str else ""
            accumulated_history_str += f"Step {step_id}:{action_str}{obs_tag}\n"

    last_turn_index = content.rfind("Now it's your turn")
    if last_turn_index != -1:
        last_section = content[last_turn_index:]
        action_match = PATTERN_FINAL_ACTION.search(last_section)
        if action_match:
            final_action = action_match.group(1).strip()
            if not steps or steps[-1]["action"] != final_action:
                steps.append(
                    {
                        "step_id": len(steps),
                        "action": final_action,
                        "memory_context": accumulated_history_str,
                        "reward": reward,
                        "done": True,
                    }
                )

    if not steps:
        single_action = PATTERN_FINAL_ACTION.search(content)
        if single_action:
            steps.append(
                {
                    "step_id": 0,
                    "action": single_action.group(1).strip(),
                    "memory_context": "",
                    "reward": reward,
                    "done": True,
                }
            )
        else:
            return None

    env_folder = os.path.basename(os.path.dirname(file_path))
    file_name = os.path.basename(file_path)

    return {
        "env_id": f"{env_folder}_{file_name}",
        "task": task_desc,
        "data_source": infer_data_source(file_path),
        "type": "all_success" if is_success else "all_fail",
        "outcome": "Success" if is_success else "Failure",
        "reward": reward,
        "trajectories": [steps],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Root containing **/*.txt rollouts")
    parser.add_argument("--output_file", required=True)
    parser.add_argument(
        "--failure_ratio",
        type=float,
        default=0.5,
        help="Per data_source: keep failures up to this fraction of successes.",
    )
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.input_dir, "**", "*.txt"), recursive=True)
    print(f"Found {len(files)} files...")

    successes_by_source, failures_by_source = {}, {}
    stats = {}

    for i, fpath in enumerate(files):
        parsed = parse_search_trajectory_file(fpath)
        if parsed is None:
            continue
        source = parsed.get("data_source", "unknown")
        stats.setdefault(source, {"Success": 0, "Failure": 0})
        stats[source][parsed["outcome"]] += 1
        target = successes_by_source if parsed["outcome"] == "Success" else failures_by_source
        target.setdefault(source, []).append(parsed)
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1} files...")

    final_dataset = []
    for source in sorted(stats):
        successes = successes_by_source.get(source, [])
        failures = sorted(failures_by_source.get(source, []), key=lambda x: x["env_id"])
        keep_failures = int(len(successes) * args.failure_ratio)
        final_dataset.extend(successes)
        final_dataset.extend(failures[:keep_failures])

    final_dataset.sort(key=lambda x: x["env_id"])
    n_success = sum(1 for e in final_dataset if e["outcome"] == "Success")
    n_fail = sum(1 for e in final_dataset if e["outcome"] == "Failure")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(final_dataset)} entries (Success={n_success}, Fail={n_fail}).")
    if stats:
        print("Per data_source:")
        for source in sorted(stats):
            s = len(successes_by_source.get(source, []))
            kept = int(s * args.failure_ratio)
            print(f"  {source}: Success {s}, Fail kept {kept}")


if __name__ == "__main__":
    main()
