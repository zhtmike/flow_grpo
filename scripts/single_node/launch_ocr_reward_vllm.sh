#!/usr/bin/env bash
# Launch a single-GPU vLLM OpenAI-compatible server hosting the GRM model
# used by the OCR reward (mirrors verl-omni's reward_model launcher).
#
# Usage:
#   bash scripts/single_node/launch_ocr_reward_vllm.sh
#
# Environment overrides:
#   REWARD_MODEL              HF model id or local path (default: Qwen/Qwen3-VL-8B-Instruct)
#   REWARD_GPU                CUDA device id to bind to (default: 0)
#   REWARD_HOST               bind host                 (default: 127.0.0.1)
#   REWARD_PORT               bind port                 (default: 17140)
#   REWARD_TP                 tensor parallel           (default: 1)
#   REWARD_GPU_MEM_UTIL       gpu_memory_utilization    (default: 0.9)
#   REWARD_MAX_MODEL_LEN      vLLM max_model_len        (default: 8192)
set -eux

REWARD_MODEL=${REWARD_MODEL:-Qwen/Qwen3-VL-8B-Instruct}
REWARD_GPU=${REWARD_GPU:-4}
REWARD_HOST=${REWARD_HOST:-127.0.0.1}
REWARD_PORT=${REWARD_PORT:-17140}
REWARD_TP=${REWARD_TP:-1}
REWARD_GPU_MEM_UTIL=${REWARD_GPU_MEM_UTIL:-0.9}
REWARD_MAX_MODEL_LEN=${REWARD_MAX_MODEL_LEN:-8192}

CUDA_VISIBLE_DEVICES=${REWARD_GPU} \
python -m vllm.entrypoints.openai.api_server \
    --model "${REWARD_MODEL}" \
    --served-model-name "${REWARD_MODEL}" \
    --host "${REWARD_HOST}" \
    --port "${REWARD_PORT}" \
    --tensor-parallel-size "${REWARD_TP}" \
    --gpu-memory-utilization "${REWARD_GPU_MEM_UTIL}" \
    --max-model-len "${REWARD_MAX_MODEL_LEN}" \
    --dtype bfloat16 \
    --trust-remote-code \
    --limit-mm-per-prompt '{"image": 1}' \
    --api-key flowgrpo > server.log 2>&1 &
