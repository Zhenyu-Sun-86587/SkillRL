"""
Detect and remove cyclic action patterns from parsed trajectories.

Some rollouts loop on the same action sequence (the model gets stuck).
This script finds the longest pattern at the trajectory tail that exactly
repeats the preceding window, and trims one repetition. It runs greedily
across all steps and works on JSON produced by parse_alfworld.py / parse_webshop.py.

Usage:
    python dedupe_repetitions.py \\
        --input_file processed_trajectories_alfworld.json \\
        --output_file processed_trajectories_alfworld_cleaned.json
"""
import argparse
import json
import os


def remove_repetitions(trajectory):
    cleaned = []
    for step in trajectory:
        cleaned.append(step)
        n = len(cleaned)
        for pattern_len in range(1, n // 2 + 1):
            last = [s.get("action") for s in cleaned[-pattern_len:]]
            prev = [s.get("action") for s in cleaned[-2 * pattern_len : -pattern_len]]
            if last == prev:
                del cleaned[-pattern_len:]
                break
    return cleaned


def process(data):
    n_modified = 0
    for entry in data:
        if "trajectories" in entry:
            new_trajs = []
            for traj in entry["trajectories"]:
                original_len = len(traj)
                cleaned = remove_repetitions(traj)
                new_trajs.append(cleaned)
                if original_len > len(cleaned):
                    n_modified += 1
            entry["trajectories"] = new_trajs
    print(f"Trajectories trimmed: {n_modified}")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = process(data)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"Saved to {args.output_file}")


if __name__ == "__main__":
    main()
