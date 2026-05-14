#!/usr/bin/env bash
# Benchmark launcher: flow_grpo Qwen-Image OCR + LoRA + async vLLM GRM reward.
# Mirrors verl-omni's `examples/flowgrpo_trainer/run_qwen_image_ocr_lora_async_reward.sh`.
#
# Step 1 (in another terminal): start the GRM vLLM server on a single GPU.
#     bash scripts/single_node/launch_ocr_reward_vllm.sh
#   (set REWARD_MODEL / REWARD_PORT / REWARD_GPU env vars to override defaults)
#
# Step 2 (this script): launch 4-GPU FSDP training that hits the GRM server async.
#
# Environment overrides:
#   ACTOR_GPUS              CUDA devices for the actor/rollout (default 0,1,2,3)
#   NUM_GPUS_ACTOR_ROLLOUT  number of training GPUs (default 4 -- must match ACTOR_GPUS)
#   MASTER_PORT             torchrun master port (default 19501)
#   REWARD_ROUTER_ADDRESS   GRM router host:port (default 127.0.0.1:17140)
#   REWARD_MODEL_NAME       served-model-name to pass to GRM (default Qwen/Qwen3-VL-8B-Instruct)
set -eux

ACTOR_GPUS=${ACTOR_GPUS:-0,1,2,3}
NUM_GPUS_ACTOR_ROLLOUT=${NUM_GPUS_ACTOR_ROLLOUT:-4}
MASTER_PORT=${MASTER_PORT:-19501}

export REWARD_ROUTER_ADDRESS=${REWARD_ROUTER_ADDRESS:-127.0.0.1:17140}
export REWARD_MODEL_NAME=${REWARD_MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}
export REWARD_API_KEY=${REWARD_API_KEY:-flowgrpo}

# Optional: set the wandb experiment name to match verl-omni for direct comparison.
export WANDB_PROJECT=${WANDB_PROJECT:-flow_grpo}
export WANDB_NAME=${WANDB_NAME:-qwen_image_ocr_lora_async_reward_flowgrpo}

CUDA_VISIBLE_DEVICES=${ACTOR_GPUS} \
torchrun --standalone \
    --nproc_per_node=${NUM_GPUS_ACTOR_ROLLOUT} \
    --master_port=${MASTER_PORT} \
    scripts/train_qwenimage.py \
    --config config/grpo.py:qwenimage_ocr_lora_async_reward_4gpu \
    "$@" > run.log 2>&1
