#!/bin/bash
# Start vLLM server WITH qwen3_xml tool-call parser (for side-by-side comparison).
# Runs on GPU 1, port 8001.
# Usage: bash eval/swe_bench/start_vllm_server_with_parser.sh

MODEL_PATH="/home/agoswami/_hackerreborn/aifsdk/_ckpts/Qwen3-14B-modified"
PORT=8001

CUDA_VISIBLE_DEVICES=1 vllm serve "$MODEL_PATH" \
  --served-model-name vllm_hosted_model \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --port "$PORT" \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --disable-log-requests
