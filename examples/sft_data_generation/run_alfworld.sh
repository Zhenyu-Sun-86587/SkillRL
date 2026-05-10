#!/usr/bin/env bash
# End-to-end ALFWorld SFT data pipeline.
#
# Input:  Directory of rollout txt files. Run your own model on ALFWorld
#         and dump trajectories first; format is documented in README.md.
# Output: ${WORK_DIR}/alfworld_sft_data.json
#         (matches https://huggingface.co/datasets/Jianwen/SkillRL-SFT-Data)
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-$(pwd)/runs/alfworld}"
ROLLOUT_DIR="${ROLLOUT_DIR:?Set ROLLOUT_DIR to your rollout txt directory}"

mkdir -p "${WORK_DIR}"

echo "==> [1/4] Parse rollouts"
python "${DIR}/preprocess/parse_alfworld.py" \
  --input_dir "${ROLLOUT_DIR}" \
  --output_file "${WORK_DIR}/processed_trajectories.json"

echo "==> [1/4] Dedupe loops"
python "${DIR}/preprocess/dedupe_repetitions.py" \
  --input_file "${WORK_DIR}/processed_trajectories.json" \
  --output_file "${WORK_DIR}/processed_trajectories_cleaned.json"

echo "==> [2/4] Generate per-trajectory skill memories"
python "${DIR}/skill_memory/generate_memory_alfworld.py" \
  --input_file "${WORK_DIR}/processed_trajectories_cleaned.json" \
  --output_file "${WORK_DIR}/generated_memories.json"

echo "==> [2/4] Aggregate memories into a skill bank"
python "${DIR}/skill_memory/aggregate_skills.py" \
  --input_file "${WORK_DIR}/generated_memories.json" \
  --output_file "${WORK_DIR}/skill_bank.json" \
  --env alfworld

echo "==> [3/4] Distill with o3"
python "${DIR}/distillation/distill_alfworld.py" \
  --input_file "${WORK_DIR}/processed_trajectories_cleaned.json" \
  --skill_bank_file "${WORK_DIR}/skill_bank.json" \
  --output_file "${WORK_DIR}/distilled_trajectories.json" \
  --model "${DISTILL_MODEL:-o3}"

echo "==> [4/4] Flatten to alpaca pairs"
python "${DIR}/postprocess/sharegpt_to_pairs.py" \
  --input_file "${WORK_DIR}/distilled_trajectories.json" \
  --output_file "${WORK_DIR}/alfworld_sft_data.json"

echo
echo "Done. SFT data: ${WORK_DIR}/alfworld_sft_data.json"
