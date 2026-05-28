"""Filter ALFWorld Alpaca SFT pairs by admissible actions.

Input records must follow the final SFT schema:
    {"instruction": "... admissible actions ...", "output": "<think>...</think>\n<action>...</action>"}

The script keeps only examples whose output action appears in the current
instruction's admissible action list.  This is useful when successful rollout
trajectories include intermediate invalid actions such as "Nothing happens".
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)
ADMISSIBLE_RE = re.compile(
    r"Your admissible actions of the current situation are:\s*\[(.*?)\]",
    re.DOTALL,
)


def parse_action(output: str) -> str:
    match = ACTION_RE.search(output or "")
    return match.group(1).strip() if match else ""


def parse_admissible_actions(instruction: str) -> list[str]:
    match = ADMISSIBLE_RE.search(instruction or "")
    if not match:
        return []
    raw = match.group(1).replace("\n", " ")
    return [item.strip().strip("'\"") for item in raw.split(",") if item.strip()]


def is_valid_pair(record: dict, allow_done: bool) -> tuple[bool, str, list[str]]:
    action = parse_action(record.get("output", ""))
    admissible_actions = parse_admissible_actions(record.get("instruction", ""))

    # ALFWorld 的 terminal turn 是后处理合成的；保留 done 可避免误删成功收尾样本。
    if allow_done and action == "done":
        return True, action, admissible_actions

    return action in set(admissible_actions), action, admissible_actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument(
        "--report_file",
        default=None,
        help="Optional JSON report path. Defaults to <output_file>.report.json",
    )
    parser.add_argument(
        "--max_report_examples",
        type=int,
        default=30,
        help="Maximum dropped examples to include in the report.",
    )
    parser.add_argument(
        "--strict_done",
        action="store_true",
        help="Require synthetic done to appear in admissible actions instead of always keeping it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    report_path = Path(args.report_file) if args.report_file else output_path.with_suffix(
        output_path.suffix + ".report.json"
    )

    with input_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    kept = []
    dropped_examples = []
    drop_reasons = {"missing_action": 0, "missing_admissible": 0, "not_admissible": 0}
    allow_done = not args.strict_done

    for idx, record in enumerate(records):
        valid, action, admissible_actions = is_valid_pair(record, allow_done=allow_done)
        if valid:
            kept.append(record)
            continue

        if not action:
            reason = "missing_action"
        elif not admissible_actions:
            reason = "missing_admissible"
        else:
            reason = "not_admissible"
        drop_reasons[reason] += 1

        if len(dropped_examples) < args.max_report_examples:
            dropped_examples.append(
                {
                    "index": idx,
                    "reason": reason,
                    "action": action,
                    "admissible_actions_head": admissible_actions[:20],
                }
            )

    report = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "total": len(records),
        "kept": len(kept),
        "dropped": len(records) - len(kept),
        "drop_ratio": (len(records) - len(kept)) / len(records) if records else 0.0,
        "drop_reasons": drop_reasons,
        "allow_done": allow_done,
        "dropped_examples": dropped_examples,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)

    os.makedirs(os.path.dirname(os.path.abspath(report_path)) or ".", exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Total: {report['total']}")
    print(f"Kept: {report['kept']}")
    print(f"Dropped: {report['dropped']} ({report['drop_ratio']:.2%})")
    print(f"Report: {report_path}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
