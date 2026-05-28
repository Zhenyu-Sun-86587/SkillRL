"""Merge ALFWorld rollout folders and renumber env directories.

The parser treats each envXXX directory as one task with multiple rollouts.
When several rollout runs are produced independently, their env ids usually
start from env000 again.  This script copies them into a new folder with
continuous env ids so the downstream parser can scan one clean directory.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

TASK_RE = re.compile(r"Your task is to:\s*(.*?)(?:\n|$)", re.IGNORECASE)
ENV_RE = re.compile(r"env(\d+)$")


def env_sort_key(path: Path) -> tuple[int, str]:
    match = ENV_RE.match(path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**12, path.name


def extract_task(file_path: Path) -> str:
    text = file_path.read_text(encoding="utf-8", errors="replace")
    match = TASK_RE.search(text)
    return match.group(1).strip() if match else ""


def collect_env_dirs(input_dir: Path) -> list[Path]:
    return sorted(
        [p for p in input_dir.iterdir() if p.is_dir() and ENV_RE.match(p.name)],
        key=env_sort_key,
    )


def validate_task_consistency(env_dir: Path, traj_files: list[Path]) -> str:
    tasks = [extract_task(path) for path in traj_files]
    unique_tasks = sorted({task for task in tasks if task})
    if len(unique_tasks) > 1:
        joined = " | ".join(unique_tasks[:5])
        raise ValueError(f"{env_dir} has multiple task descriptions: {joined}")
    return unique_tasks[0] if unique_tasks else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        required=True,
        help="Rollout roots containing envXXX/test*.txt directories.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New merged rollout root. It must not already contain files.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--min-trajs-per-env",
        type=int,
        default=3,
        help="Skip envs with fewer than this many test*.txt files.",
    )
    parser.add_argument(
        "--skip-mismatched-task-envs",
        action="store_true",
        help="Skip env dirs whose test files are not the same task.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dirs = [Path(p).expanduser().resolve() for p in args.input_dirs]
    output_dir = Path(args.output_dir).expanduser().resolve()

    for input_dir in input_dirs:
        if not input_dir.exists():
            raise FileNotFoundError(f"Input dir does not exist: {input_dir}")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output dir is not empty: {output_dir}. Use a new directory to keep raw rollouts intact."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    next_env_id = args.start_index
    skipped = 0

    for input_dir in input_dirs:
        for env_dir in collect_env_dirs(input_dir):
            traj_files = sorted(env_dir.glob("test*.txt"))
            if len(traj_files) < args.min_trajs_per_env:
                skipped += 1
                continue

            try:
                task = validate_task_consistency(env_dir, traj_files)
            except ValueError:
                if not args.skip_mismatched_task_envs:
                    raise
                skipped += 1
                continue

            # 只复制 test*.txt，避免把旧的 processed.json 或临时文件混进新目录。
            dst_env_name = f"env{next_env_id:03d}"
            dst_env_dir = output_dir / dst_env_name
            dst_env_dir.mkdir()
            for src_file in traj_files:
                shutil.copy2(src_file, dst_env_dir / src_file.name)

            manifest.append(
                {
                    "merged_env_id": dst_env_name,
                    "source_dir": str(input_dir),
                    "source_env_id": env_dir.name,
                    "num_trajectories": len(traj_files),
                    "task": task,
                }
            )
            next_env_id += 1

    manifest_path = output_dir / "merge_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Merged {len(manifest)} envs into {output_dir}")
    print(f"Skipped {skipped} envs")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
