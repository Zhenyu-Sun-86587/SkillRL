#!/usr/bin/env bash
# 下载 SkillRL embedding retrieval 默认使用的 Qwen3-Embedding-0.6B。
# 默认保存到项目 models 目录，便于 RL 阶段通过本地路径加载 embedding 模型。
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-Embedding-0.6B}"
MODEL_DIR="${MODEL_DIR:-/home/sunzhengyu/SkillRL/models/Qwen3-Embedding-0.6B}"
export MODEL_ID MODEL_DIR

mkdir -p "${MODEL_DIR}"

if command -v hf >/dev/null 2>&1; then
  hf download "${MODEL_ID}" \
    --local-dir "${MODEL_DIR}" \
    --max-workers "${HF_MAX_WORKERS:-8}"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "${MODEL_ID}" \
    --local-dir "${MODEL_DIR}"
else
  python - <<'PY'
import os
from huggingface_hub import snapshot_download

model_id = os.environ.get("MODEL_ID", "Qwen/Qwen3-Embedding-0.6B")
model_dir = os.environ.get("MODEL_DIR", "/home/sunzhengyu/SkillRL/models/Qwen3-Embedding-0.6B")

# snapshot_download 支持断点续传；local_dir 固定到项目目录，避免模型散落在 HF cache。
snapshot_download(
    repo_id=model_id,
    local_dir=model_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)
PY
fi

echo "Downloaded ${MODEL_ID} to ${MODEL_DIR}"
