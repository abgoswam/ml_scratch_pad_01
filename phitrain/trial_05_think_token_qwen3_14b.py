"""
Test whether enable_thinking=True/False affects <think> token presence
for the Qwen3-14B model served via vLLM.

This model's chat template HAS enable_thinking support:
  - When enable_thinking=False, template prepends empty <think></think> to suppress thinking
  - When enable_thinking=True (or unset), model can produce <think> blocks
  - vLLM 0.11+ also has a built-in reasoning parser for Qwen3

Usage:
  1. Start vLLM:
     CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-14B \
       --tensor-parallel-size 4 --port 8002 --served-model-name qwen3-14b

  2. Run this script:
     python trial_05_think_token_qwen3_14b.py
"""

import json
import requests
import sys
import time

VLLM_BASE = "http://localhost:8002/v1"
MODEL_NAME = "qwen3-14b"

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
    "temperature": 0.6,
    "max_tokens": 4096,
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

    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    finish = data["choices"][0].get("finish_reason", "?")

    print(f"\n--- Raw content (finish_reason={finish}) ---")
    print(content[:1000] if content else "(empty)")

    if reasoning:
        print(f"\n--- reasoning_content field ---")
        print(reasoning[:1000])

    has_think_tag = "<think>" in content or "</think>" in content
    print(f"\n--- Analysis ---")
    print(f"  Contains <think> tags in content: {has_think_tag}")
    print(f"  Has reasoning_content field: {bool(reasoning)}")
    print(f"  Content length: {len(content)} chars")
    print(f"  Reasoning length: {len(reasoning)} chars")


def main() -> None:
    print("=" * 60)
    print("MODEL: Qwen3-14B (original, NOT 2507)")
    print("Chat template HAS enable_thinking support.")
    print("enable_thinking=False -> prepends empty <think></think>")
    print("vLLM may also use built-in reasoning parser to split")
    print("thinking into reasoning_content field.")
    print("=" * 60)

    if not wait_for_server():
        print("[ERROR] vLLM server not running. Start with:")
        print(
            "  CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-14B "
            "--tensor-parallel-size 4 --port 8002 --served-model-name qwen3-14b"
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

    # Test 4: enable_thinking=False as top-level param
    run_test(
        "enable_thinking=False (top-level param)",
        {"enable_thinking": False},
    )

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(
        "Qwen3-14B chat template supports enable_thinking natively.\n"
        "Expected behavior:\n"
        "  - Baseline / enable_thinking=True: model produces <think> blocks\n"
        "    (vLLM may split into reasoning_content field)\n"
        "  - enable_thinking=False: template suppresses thinking,\n"
        "    model should skip or produce empty <think> blocks\n"
        "\n"
        "Compare with omniagent-v1 (hardcoded, no param support) and\n"
        "Qwen3-4B-Instruct-2507 (no thinking at all)."
    )


if __name__ == "__main__":
    main()
