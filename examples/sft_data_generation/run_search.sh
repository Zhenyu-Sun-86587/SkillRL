#!/usr/bin/env bash
# End-to-end Search-agent SFT data pipeline.
#
# Input:  Directory of rollout txt files (run your own model first).
# Output: ${WORK_DIR}/search_sft_data.json
#         (matches https://huggingface.co/datasets/Jianwen/SkillRL-SFT-Data)
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-$(pwd)/runs/search}"
ROLLOUT_DIR="${ROLLOUT_DIR:?Set ROLLOUT_DIR to your rollout txt directory}"

mkdir -p "${WORK_DIR}"

echo "==> [1/4] Parse rollouts"
python "${DIR}/preprocess/parse_search.py" \
  --input_dir "${ROLLOUT_DIR}" \
  --output_file "${WORK_DIR}/processed_trajectories.json" \
  --failure_ratio 0.5

echo "==> [2/4] Generate per-trajectory skill memories"
python "${DIR}/skill_memory/generate_memory_search.py" \
  --input_file "${WORK_DIR}/processed_trajectories.json" \
  --output_file "${WORK_DIR}/generated_memories.json"

echo "==> [2/4] Aggregate memories into a skill bank"
python "${DIR}/skill_memory/aggregate_skills.py" \
  --input_file "${WORK_DIR}/generated_memories.json" \
  --output_file "${WORK_DIR}/skill_bank.json" \
  --env search

echo "==> [3/4] Distill with o3"
python "${DIR}/distillation/distill_search.py" \
  --input_file "${WORK_DIR}/processed_trajectories.json" \
  --skill_bank_file "${WORK_DIR}/skill_bank.json" \
  --output_file "${WORK_DIR}/distilled_trajectories.json" \
  --model "${DISTILL_MODEL:-o3}"

echo "==> [4/4] Flatten to alpaca pairs"
python "${DIR}/postprocess/sharegpt_to_pairs.py" \
  --input_file "${WORK_DIR}/distilled_trajectories.json" \
  --output_file "${WORK_DIR}/search_sft_data.json"

echo
echo "Done. SFT data: ${WORK_DIR}/search_sft_data.json"
