"""
Test whether enable_thinking=True/False affects <think> token presence
for the Qwen3-4B-Instruct-2507 model served via vLLM.

Usage:
  1. Start vLLM:
     CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3-4B-Instruct-2507 \
       --tensor-parallel-size 2 --port 8001 --served-model-name qwen3-4b

  2. Run this script:
     python trial_05_think_token_qwen3.py
"""

import json
import requests
import sys
import time

VLLM_BASE = "http://localhost:8001/v1"
MODEL_NAME = "qwen3-4b"

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": (
            "A farmer has 17 sheep. All but 9 run away. "
            "How many sheep does the farmer have left? "
            "Think step by step."
        ),
    },
]

COMMON_KWARGS = {
    "model": MODEL_NAME,
    "messages": MESSAGES,
    "temperature": 0.0,
    "max_tokens": 2048,
    "seed": 42,
}


def wait_for_server(timeout: int = 10) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{VLLM_BASE}/models", timeout=5)
            if r.status_code == 200:
                print(f"[OK] vLLM server ready ({time.time() - start:.0f}s)")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(2)
    return False


def run_test(label: str, extra_params: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"TEST: {label}")
    print(f"Extra params: {extra_params}")
    print("=" * 60)

    payload = {**COMMON_KWARGS, **extra_params}

    try:
        r = requests.post(f"{VLLM_BASE}/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] HTTP {e.response.status_code}: {e.response.text[:500]}")
        return
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    content = data["choices"][0]["message"]["content"] or ""
    finish = data["choices"][0].get("finish_reason", "?")

    print(f"\n--- Raw response (finish_reason={finish}) ---")
    print(content)

    has_think_tag = "<think>" in content or "</think>" in content
    print(f"\n--- Analysis ---")
    print(f"  Contains <think> tags: {has_think_tag}")
    print(f"  Response length: {len(content)} chars")


def main() -> None:
    print("=" * 60)
    print("MODEL: Qwen3-4B-Instruct-2507")
    print("Chat template has NO enable_thinking support.")
    print("No <think> instructions in template.")
    print("=" * 60)

    if not wait_for_server():
        print("[ERROR] vLLM server not running. Start with:")
        print(
            "  CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3-4B-Instruct-2507 "
            "--tensor-parallel-size 2 --port 8001 --served-model-name qwen3-4b"
        )
        sys.exit(1)

    # Test 1: baseline (no enable_thinking param)
    run_test("Baseline (no enable_thinking param)", {})

    # Test 2: enable_thinking=True via chat_template_kwargs
    run_test(
        "enable_thinking=True (chat_template_kwargs)",
        {"chat_template_kwargs": {"enable_thinking": True}},
    )

    # Test 3: enable_thinking=False via chat_template_kwargs
    run_test(
        "enable_thinking=False (chat_template_kwargs)",
        {"chat_template_kwargs": {"enable_thinking": False}},
    )

    # Test 4: enable_thinking=False as top-level param (some vLLM versions)
    run_test(
        "enable_thinking=False (top-level param)",
        {"enable_thinking": False},
    )

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(
        "Qwen3-4B-Instruct-2507 chat template does NOT reference enable_thinking.\n"
        "However, Qwen3 models are trained to produce <think> blocks by default.\n"
        "Whether vLLM strips them depends on:\n"
        "  1. Whether vLLM has built-in Qwen3 thinking support\n"
        "  2. Whether the template processes the param (it doesn't in 2507)\n"
        "\n"
        "Compare the results above to see if <think> tags appear/disappear."
    )


if __name__ == "__main__":
    main()
