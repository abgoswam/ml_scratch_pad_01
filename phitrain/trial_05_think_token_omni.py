"""
Test whether enable_thinking=True/False affects <think> token presence
for the omniagent-v1 model served via vLLM.

Uses the actual swebench system prompt (custom, not the hardcoded one)
to verify that the model produces <think> from SFT weights regardless
of system prompt, and that enable_thinking=False suppresses it.

Usage:
  1. Start vLLM:
     CUDA_VISIBLE_DEVICES=0,1 vllm serve /home/agoswami/_hackerreborn/aifsdk/_ckpts/omniagent-v1 \
       --tensor-parallel-size 2 --port 8000 --served-model-name omniagent-v1

  2. Run this script:
     python trial_05_think_token_omni.py
"""

import json
import requests
import sys
import time

VLLM_BASE = "http://localhost:8000/v1"
MODEL_NAME = "omniagent-v1"

SWEBENCH_SYSTEM_PROMPT = (
    "You are a helpful assistant that can interact multiple times with a "
    "computer shell to solve programming tasks.\n"
    "Your response must contain exactly ONE bash code block with ONE command "
    "(or commands connected with && or ||).\n"
    "\n"
    "Include a THOUGHT section before your command where you explain your "
    "reasoning process.\n"
    "Format your response as shown in <format_example>.\n"
    "\n"
    "<format_example>\n"
    "THOUGHT: Your reasoning and analysis here\n"
    "\n"
    "```mswea_bash_command\n"
    "your_command_here\n"
    "```\n"
    "</format_example>\n"
    "\n"
    "Failure to follow these rules will cause your response to be rejected."
)

MESSAGES = [
    {"role": "system", "content": SWEBENCH_SYSTEM_PROMPT},
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

    print(f"\n--- Raw response (finish_reason={finish}, first 1000 chars) ---")
    print(content[:1000])

    has_think_tag = "<think>" in content or "</think>" in content
    print(f"\n--- Analysis ---")
    print(f"  Contains <think> tags: {has_think_tag}")
    print(f"  Response length: {len(content)} chars")


def main() -> None:
    print("=" * 60)
    print("MODEL: omniagent-v1 (base SFT checkpoint)")
    print("Using SWEBENCH system prompt (custom, not hardcoded).")
    print("Model produces <think> from SFT weights regardless of prompt.")
    print("=" * 60)

    if not wait_for_server():
        print("[ERROR] vLLM server not running. Start with:")
        print(
            "  CUDA_VISIBLE_DEVICES=0,1 vllm serve "
            "/home/agoswami/_hackerreborn/aifsdk/_ckpts/omniagent-v1 "
            "--tensor-parallel-size 2 --port 8000 --served-model-name omniagent-v1"
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
        "omniagent-v1 produces <think> from SFT weights (not just system prompt).\n"
        "With enable_thinking support added to the chat template:\n"
        "  - Baseline / enable_thinking=True: model should produce <think> blocks\n"
        "  - enable_thinking=False: template prepends empty <think></think>,\n"
        "    suppressing thinking (same mechanism as Qwen3-14B)"
    )


if __name__ == "__main__":
    main()
