#!/bin/bash
# Start vLLM server for local single-instance SWE-bench debugging.
# Usage: bash eval/swe_bench/start_vllm_server.sh

MODEL_PATH="/home/agoswami/_hackerreborn/aifsdk/_ckpts/Qwen3-14B-modified"
PORT=8000

CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_PATH" \
  --served-model-name vllm_hosted_model \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --port "$PORT" \
  --trust-remote-code \
  --disable-log-requests
