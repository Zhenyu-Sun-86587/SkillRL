"""
Parse raw ALFWorld rollout trajectories (txt format) into structured JSON.

Input:  Directory containing env*/test*.txt rollouts produced by 01_rollout.
Output: JSON with one record per env, each holding multiple trajectories.

Filtering:
- Successful trajectories: identified by 'Reward: 10.000'.
- Per env, we sort trajectories by length (shortest first) and keep:
    * up to N_SUCCESS trajectories from "all-success" envs
    * up to N_FAIL  trajectories from "all-fail"   envs
- Mixed envs (some success + some fail) are skipped by default; uncomment
  the 'mixed' branch in main() to include them.

Usage:
    python parse_alfworld.py \\
        --input_dir /path/to/trajectories_qwen2.5 \\
        --output_file processed_trajectories_alfworld.json \\
        --existing_file processed_trajectories_alfworld_prev.json  # optional
"""
import argparse
import glob
import json
import os
import re

PATTERN_REASONING = re.compile(
    r"You should first reason step-by-step about the current situation\. "
    r"This reasoning process MUST be enclosed within <think> </think> tags\.\s*",
    re.DOTALL,
)
PATTERN_WELCOME = re.compile(r"-=\s*Welcome to TextWorld, ALFRED!\s*=-", re.IGNORECASE)
PATTERN_TURN = re.compile(
    r"Now it's your turn to take an action\.\s*"
    r"Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags\.(\s*)",
    re.DOTALL,
)
PATTERN_ADMISSIBLE_BLOCK = re.compile(
    r"Your admissible actions of the current situation are:\s*\[(.*?)\]\.", re.DOTALL
)
PATTERN_OBS_PREFIX = re.compile(
    r"(?:You are now at step \d+ and your current observation is:|Your current observation is:)\s*",
    re.IGNORECASE,
)
PATTERN_TASK_EXTRACT = re.compile(r"Your task is to:\s*(.*?)(?:\n|$)", re.IGNORECASE)
PATTERN_TASK_INLINE = re.compile(r"\n*Your task is to:\s*.*?(?:\n|$)", re.IGNORECASE)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = PATTERN_REASONING.sub("", text)
    text = PATTERN_WELCOME.sub("", text)
    text = PATTERN_TURN.sub("", text)
    text = PATTERN_TASK_INLINE.sub("", text)
    return text.strip()


def parse_trajectory_file(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    task_match = PATTERN_TASK_EXTRACT.search(content)
    task_desc = task_match.group(1).strip() if task_match else ""

    is_success = "Reward: 10.000" in content
    raw_steps = re.split(r"\n(?=Step -?\d+ \|)", content)
    parsed_steps = []

    for segment in raw_steps:
        if not segment.strip():
            continue

        header_match = re.match(
            r"(Step -?\d+) \| Action: (.*?) \| Reward: ([\d\.]+) \| Done: (True|False)",
            segment,
        )
        if not header_match:
            continue

        step_id = header_match.group(1)
        action = header_match.group(2)
        reward = float(header_match.group(3))
        done = header_match.group(4) == "True"

        obs_raw = segment[header_match.end() :].strip()
        if obs_raw.startswith("Obs:"):
            obs_raw = obs_raw[4:].strip()

        admissible_actions = []
        adm_match = PATTERN_ADMISSIBLE_BLOCK.search(obs_raw)
        if adm_match:
            actions_str = adm_match.group(1)
            for act in actions_str.split("\n"):
                act = act.strip()
                if not act:
                    continue
                if act.endswith(","):
                    act = act[:-1].strip()
                if (act.startswith("'") and act.endswith("'")) or (
                    act.startswith('"') and act.endswith('"')
                ):
                    act = act[1:-1]
                admissible_actions.append(act)
            obs_raw = obs_raw.replace(adm_match.group(0), "")

        prefix_match = PATTERN_OBS_PREFIX.search(obs_raw)
        obs_content = obs_raw[prefix_match.end() :] if prefix_match else obs_raw
        obs_clean = clean_text(obs_content)

        parsed_steps.append(
            {
                "step_id": step_id,
                "action": action,
                "reward": reward,
                "done": done,
                "observation": obs_clean,
                "admissible_actions": admissible_actions,
            }
        )

    return {
        "is_success": is_success,
        "length": len(parsed_steps),
        "steps": parsed_steps,
        "task_desc": task_desc,
        "file_path": file_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Root with env*/test*.txt rollouts")
    parser.add_argument("--output_file", required=True)
    parser.add_argument(
        "--existing_file",
        default=None,
        help="Skip envs already present in this JSON (used for incremental builds)",
    )
    parser.add_argument("--n_success_envs", type=int, default=70)
    parser.add_argument("--n_fail_envs", type=int, default=30)
    parser.add_argument("--n_trajs_per_env", type=int, default=3)
    parser.add_argument(
        "--min_trajs_per_env",
        type=int,
        default=3,
        help="Skip envs with fewer than this many trajectory files",
    )
    args = parser.parse_args()

    skip_envs = set()
    if args.existing_file and os.path.exists(args.existing_file):
        with open(args.existing_file, "r", encoding="utf-8") as f:
            for entry in json.load(f):
                skip_envs.add(entry["env_id"])

    env_dirs = sorted(glob.glob(os.path.join(args.input_dir, "env*")))
    print(f"Scanning {len(env_dirs)} env dirs in {args.input_dir} ...")

    all_envs_data = {}
    for env_path in env_dirs:
        env_name = os.path.basename(env_path)
        if env_name in skip_envs:
            continue

        traj_files = glob.glob(os.path.join(env_path, "*.txt"))
        if len(traj_files) < args.min_trajs_per_env:
            continue

        success_list, fail_list = [], []
        common_task_desc = ""
        for t_file in traj_files:
            parsed = parse_trajectory_file(t_file)
            if not common_task_desc and parsed["task_desc"]:
                common_task_desc = parsed["task_desc"]
            (success_list if parsed["is_success"] else fail_list).append(parsed)

        success_list.sort(key=lambda x: x["length"])
        all_envs_data[env_name] = {
            "success": success_list,
            "fail": fail_list,
            "task": common_task_desc,
        }

    all_success_envs, all_fail_envs, mixed_envs = [], [], []
    for name, d in all_envs_data.items():
        s, f = len(d["success"]), len(d["fail"])
        if s >= 2 and f >= 1:
            mixed_envs.append(name)
        elif s >= args.n_trajs_per_env:
            all_success_envs.append(name)
        elif f >= args.n_trajs_per_env:
            all_fail_envs.append(name)

    print(
        f"Mixed={len(mixed_envs)} | All-Success={len(all_success_envs)} | "
        f"All-Fail={len(all_fail_envs)}"
    )

    selected_success = all_success_envs[: args.n_success_envs]
    selected_fail = all_fail_envs[: args.n_fail_envs]

    final_dataset = []
    for env_name in selected_success:
        picks = all_envs_data[env_name]["success"][: args.n_trajs_per_env]
        final_dataset.append(
            {
                "env_id": env_name,
                "task": all_envs_data[env_name]["task"],
                "type": "all_success",
                "trajectories": [p["steps"] for p in picks],
            }
        )
    for env_name in selected_fail:
        picks = all_envs_data[env_name]["fail"][: args.n_trajs_per_env]
        final_dataset.append(
            {
                "env_id": env_name,
                "task": all_envs_data[env_name]["task"],
                "type": "all_fail",
                "trajectories": [p["steps"] for p in picks],
            }
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(final_dataset)} envs to {args.output_file}")


if __name__ == "__main__":
    main()
