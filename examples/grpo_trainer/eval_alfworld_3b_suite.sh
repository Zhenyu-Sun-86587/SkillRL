#!/usr/bin/env bash
set -uo pipefail

# Run the ALFWorld evaluation matrix for the 3B SkillRL reproduction.
# This wrapper is intentionally thin: it delegates all model/env logic to the
# existing run_alfworld_skills.sh and only varies model, checkpoint, skill bank,
# and seed.

ROOT_DIR="${ROOT_DIR:-/home/sunzhengyu/SkillRL}"
RUNNER="${ROOT_DIR}/examples/grpo_trainer/run_alfworld_skills.sh"
ENGINE="${ENGINE:-vllm}"

SEEDS="${SEEDS:-1 2 3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,5}"
N_GPUS="${N_GPUS:-2}"
TP_SIZE="${TP_SIZE:-2}"

MODEL_BASE="${MODEL_BASE:-${ROOT_DIR}/models/Qwen2.5-3B-Instruct}"
MODEL_SFT="${MODEL_SFT:-${ROOT_DIR}/models/Qwen2.5-3B-Instruct-SkillRL-SFT-valid}"

EMBEDDING_CKPT_DIR="${EMBEDDING_CKPT_DIR:-${ROOT_DIR}/checkpoints/verl_agent_alfworld/grpo_qwen2.5_3b_sft_embedding_skills}"
EMBEDDING_STEP="${EMBEDDING_STEP:-60}"
EMBEDDING_UPDATED_SKILLS="${EMBEDDING_UPDATED_SKILLS:-${EMBEDDING_CKPT_DIR}/updated_skills_step50.json}"
EMBEDDING_ORIGIN_SKILLS="${EMBEDDING_ORIGIN_SKILLS:-${ROOT_DIR}/runs/sft_data/alfworld_qwen25_3b_deepseek/skill_bank.json}"

CLAUDE_CKPT_DIR="${CLAUDE_CKPT_DIR:-${ROOT_DIR}/checkpoints/verl_agent_alfworld/grpo_qwen2.5_3b_sft_claude_style_skills}"
CLAUDE_STEP="${CLAUDE_STEP:-55}"
CLAUDE_UPDATED_SKILLS="${CLAUDE_UPDATED_SKILLS:-${CLAUDE_CKPT_DIR}/updated_skills_step40.json}"
CLAUDE_ORIGIN_SKILLS="${CLAUDE_ORIGIN_SKILLS:-${ROOT_DIR}/memory_data/alfworld/claude_style_skills.json}"

OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/eval_alfworld_3b_suite/$(date +%Y%m%d_%H%M%S)}"
RAY_TMPDIR="${RAY_TMPDIR:-${ROOT_DIR}/ray_tmp}"
TMPDIR="${TMPDIR:-${RAY_TMPDIR}}"
STOP_RAY_BETWEEN_RUNS="${STOP_RAY_BETWEEN_RUNS:-1}"
ENABLE_WANDB="${ENABLE_WANDB:-0}"

mkdir -p "${OUT_ROOT}" "${RAY_TMPDIR}"
export CUDA_VISIBLE_DEVICES
export RAY_TMPDIR
export TMPDIR

if [[ ! -f "${RUNNER}" ]]; then
  echo "Runner not found: ${RUNNER}" >&2
  exit 1
fi

if [[ "${ENABLE_WANDB}" == "1" ]]; then
  LOGGER_OVERRIDE="trainer.logger=['console','wandb']"
else
  LOGGER_OVERRIDE="trainer.logger=['console']"
fi

COMMON_EVAL_OVERRIDES=(
  "trainer.val_before_train=True"
  "trainer.val_only=True"
  "trainer.n_gpus_per_node=${N_GPUS}"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE}"
  "actor_rollout_ref.actor.ppo_mini_batch_size=32"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.rollout.gpu_memory_utilization=0.35"
  "actor_rollout_ref.rollout.max_num_batched_tokens=8192"
  "actor_rollout_ref.rollout.max_num_seqs=64"
  "actor_rollout_ref.rollout.enforce_eager=True"
  "actor_rollout_ref.rollout.free_cache_engine=True"
  "actor_rollout_ref.rollout.val_kwargs.n=1"
  "env.rollout.n=4"
  "${LOGGER_OVERRIDE}"
)

run_eval() {
  local exp_name="$1"
  local seed="$2"
  local model_path="$3"
  local resume_mode="$4"
  local resume_path="$5"
  local checkpoint_dir="$6"
  local skills_path="$7"
  local use_skills="$8"

  local run_dir="${OUT_ROOT}/${exp_name}/seed_${seed}"
  local log_file="${run_dir}/eval.log"
  local cmd_file="${run_dir}/command.sh"
  local metrics_file="${run_dir}/metrics.json"
  mkdir -p "${run_dir}"

  local cmd=(
    "bash" "${RUNNER}" "${ENGINE}"
    "${COMMON_EVAL_OVERRIDES[@]}"
    "env.seed=${seed}"
    "actor_rollout_ref.model.path=${model_path}"
    "trainer.experiment_name=eval_${exp_name}_seed${seed}"
    "trainer.resume_mode=${resume_mode}"
    "trainer.default_local_dir=${checkpoint_dir}"
    "env.use_skills_only_memory=${use_skills}"
    "env.skills_only_memory.enable_dynamic_update=False"
  )

  if [[ "${resume_mode}" == "resume_path" ]]; then
    cmd+=("trainer.resume_from_path=${resume_path}")
  fi

  if [[ "${use_skills}" == "True" ]]; then
    cmd+=("env.skills_only_memory.skills_json_path=${skills_path}")
  fi

  {
    printf 'cd "%s"\n' "${ROOT_DIR}"
    printf 'export CUDA_VISIBLE_DEVICES="%s"\n' "${CUDA_VISIBLE_DEVICES}"
    printf 'export RAY_TMPDIR="%s"\n' "${RAY_TMPDIR}"
    printf 'export TMPDIR="%s"\n' "${TMPDIR}"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } > "${cmd_file}"

  echo "==== RUN ${exp_name} seed=${seed} ===="
  echo "log: ${log_file}"

  if [[ "${STOP_RAY_BETWEEN_RUNS}" == "1" ]]; then
    ray stop -f >/dev/null 2>&1 || true
  fi

  set +e
  (
    cd "${ROOT_DIR}" &&
    "${cmd[@]}"
  ) 2>&1 | tee "${log_file}"
  local status=${PIPESTATUS[0]}
  set -e

  python3 - "${log_file}" "${metrics_file}" "${exp_name}" "${seed}" "${status}" <<'PY'
import json
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
experiment = sys.argv[3]
seed = int(sys.argv[4])
exit_code = int(sys.argv[5])

text = log_path.read_text(encoding="utf-8", errors="ignore")
pattern = re.compile(
    r"(?P<key>(?:val|episode)/[A-Za-z0-9_./-]+)(?::|\s+)"
    r"(?P<value>[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?)"
)

metrics = {}
for match in pattern.finditer(text):
    key = match.group("key")
    try:
        metrics[key] = float(match.group("value"))
    except ValueError:
        pass

payload = {
    "experiment": experiment,
    "seed": seed,
    "exit_code": exit_code,
    "metrics": metrics,
}
out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
PY

  echo "{\"experiment\":\"${exp_name}\",\"seed\":${seed},\"exit_code\":${status},\"log\":\"${log_file}\",\"metrics\":\"${metrics_file}\"}" >> "${OUT_ROOT}/runs_manifest.jsonl"
  return 0
}

for seed in ${SEEDS}; do
  run_eval \
    "base-model-3b" \
    "${seed}" \
    "${MODEL_BASE}" \
    "disable" \
    "" \
    "${OUT_ROOT}/_scratch/base-model-3b/seed_${seed}" \
    "" \
    "False"

  run_eval \
    "cold-start-model-3b" \
    "${seed}" \
    "${MODEL_SFT}" \
    "disable" \
    "" \
    "${OUT_ROOT}/_scratch/cold-start-model-3b/seed_${seed}" \
    "" \
    "False"

  run_eval \
    "embedding-skill-rl-3b-updated-skills" \
    "${seed}" \
    "${MODEL_SFT}" \
    "resume_path" \
    "${EMBEDDING_CKPT_DIR}/global_step_${EMBEDDING_STEP}" \
    "${EMBEDDING_CKPT_DIR}" \
    "${EMBEDDING_UPDATED_SKILLS}" \
    "True"

  run_eval \
    "embedding-skill-rl-3b-without-updated-skills" \
    "${seed}" \
    "${MODEL_SFT}" \
    "resume_path" \
    "${EMBEDDING_CKPT_DIR}/global_step_${EMBEDDING_STEP}" \
    "${EMBEDDING_CKPT_DIR}" \
    "${EMBEDDING_ORIGIN_SKILLS}" \
    "True"

  run_eval \
    "claude-style-skill-rl-3b-updated-skills" \
    "${seed}" \
    "${MODEL_SFT}" \
    "resume_path" \
    "${CLAUDE_CKPT_DIR}/global_step_${CLAUDE_STEP}" \
    "${CLAUDE_CKPT_DIR}" \
    "${CLAUDE_UPDATED_SKILLS}" \
    "True"

  run_eval \
    "claude-style-skill-rl-3b-without-updated-skills" \
    "${seed}" \
    "${MODEL_SFT}" \
    "resume_path" \
    "${CLAUDE_CKPT_DIR}/global_step_${CLAUDE_STEP}" \
    "${CLAUDE_CKPT_DIR}" \
    "${CLAUDE_ORIGIN_SKILLS}" \
    "True"

done

python3 - "${OUT_ROOT}" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
records = []
metric_keys = set()

for path in sorted(root.glob("*/seed_*/metrics.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    metric_keys.update(metrics)
    row = {
        "experiment": data["experiment"],
        "seed": data["seed"],
        "exit_code": data["exit_code"],
        **metrics,
    }
    records.append(row)

preferred = [
    "val/success_rate",
    "val/text/test_score",
    "val/pick_and_place_success_rate",
    "val/look_at_obj_in_light_success_rate",
    "val/pick_clean_then_place_in_recep_success_rate",
    "val/pick_heat_then_place_in_recep_success_rate",
    "val/pick_cool_then_place_in_recep_success_rate",
    "val/pick_two_obj_and_place_success_rate",
]
metric_order = [k for k in preferred if k in metric_keys] + sorted(metric_keys - set(preferred))

with (root / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["experiment", "seed", "exit_code", *metric_order])
    writer.writeheader()
    for row in records:
        writer.writerow(row)

summary_rows = []
for exp in sorted({r["experiment"] for r in records}):
    exp_rows = [r for r in records if r["experiment"] == exp]
    summary = {"experiment": exp, "runs": len(exp_rows), "failed_runs": sum(r["exit_code"] != 0 for r in exp_rows)}
    for key in metric_order:
        vals = [r[key] for r in exp_rows if key in r]
        if vals:
            summary[f"{key}/mean"] = statistics.mean(vals)
            summary[f"{key}/std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    summary_rows.append(summary)

summary_fields = ["experiment", "runs", "failed_runs"]
for key in metric_order:
    summary_fields.extend([f"{key}/mean", f"{key}/std"])

with (root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=summary_fields)
    writer.writeheader()
    for row in summary_rows:
        writer.writerow(row)

lines = ["# ALFWorld 3B Evaluation Summary", ""]
lines.append(f"Output root: `{root}`")
lines.append("")
lines.append("| experiment | runs | failed | val/success_rate mean | val/success_rate std |")
lines.append("|---|---:|---:|---:|---:|")
for row in summary_rows:
    mean = row.get("val/success_rate/mean", "")
    std = row.get("val/success_rate/std", "")
    mean_s = f"{mean:.6f}" if isinstance(mean, float) else ""
    std_s = f"{std:.6f}" if isinstance(std, float) else ""
    lines.append(f"| {row['experiment']} | {row['runs']} | {row['failed_runs']} | {mean_s} | {std_s} |")

(root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"Wrote per-seed metrics: {root / 'per_seed_metrics.csv'}")
print(f"Wrote summary: {root / 'summary.csv'}")
print(f"Wrote markdown summary: {root / 'summary.md'}")
PY

echo "All done. Output root: ${OUT_ROOT}"
