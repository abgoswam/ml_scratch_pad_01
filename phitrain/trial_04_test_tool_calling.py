"""
Test whether omniagent-v1 supports native tool calling vs text-based action parsing.

Usage:
  1. Start vLLM:
     CUDA_VISIBLE_DEVICES=0,1 vllm serve /home/agoswami/_hackerreborn/aifsdk/_ckpts/omniagent-v1 \
       --tensor-parallel-size 2 --port 8000 --served-model-name omniagent-v1

  2. Run this script:
     python trial_04_test_tool_calling.py
"""

import json
import requests
import sys
import time

VLLM_BASE = "http://localhost:8000/v1"
MODEL_NAME = "omniagent-v1"

SYSTEM_MSG = (
    "You are a helpful assistant that can interact with a computer shell "
    "to solve programming tasks."
)
USER_MSG = "List all Python files in the current directory."

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}

MESSAGES = [
    {"role": "system", "content": SYSTEM_MSG},
    {"role": "user", "content": USER_MSG},
]

COMMON_KWARGS = {
    "model": MODEL_NAME,
    "messages": MESSAGES,
    "temperature": 0.0,
    "max_tokens": 512,
    "seed": 42,
}


def wait_for_server(timeout: int = 300) -> bool:
    """Wait for vLLM server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{VLLM_BASE}/models", timeout=5)
            if r.status_code == 200:
                print(f"[OK] vLLM server is ready ({time.time() - start:.0f}s)")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(5)
    return False


def test_text_based() -> None:
    """Test text-based mode (no tools, model outputs free-form text)."""
    print("\n" + "=" * 60)
    print("TEST 1: TEXT-BASED (no tools param)")
    print("=" * 60)

    payload = {**COMMON_KWARGS}
    r = requests.post(f"{VLLM_BASE}/chat/completions", json=payload)
    r.raise_for_status()
    data = r.json()

    content = data["choices"][0]["message"]["content"]
    tool_calls = data["choices"][0]["message"].get("tool_calls")

    print(f"\n--- Response content ---\n{content}")
    print(f"\n--- Tool calls ---\n{tool_calls}")

    # Check if it naturally outputs the mswea_bash_command format
    if "```mswea_bash_command" in (content or ""):
        print("\n[RESULT] Model outputs ```mswea_bash_command``` blocks naturally!")
    elif "```bash" in (content or "") or "```" in (content or ""):
        print("\n[RESULT] Model outputs generic code blocks (not mswea-specific).")
    else:
        print("\n[RESULT] Model outputs plain text, no code blocks.")


def test_tool_calling() -> None:
    """Test native tool calling mode (pass tools param)."""
    print("\n" + "=" * 60)
    print("TEST 2: NATIVE TOOL CALLING (with tools param)")
    print("=" * 60)

    payload = {**COMMON_KWARGS, "tools": [BASH_TOOL], "tool_choice": "auto"}
    try:
        r = requests.post(f"{VLLM_BASE}/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()

        msg = data["choices"][0]["message"]
        content = msg.get("content")
        tool_calls = msg.get("tool_calls")

        print(f"\n--- Response content ---\n{content}")
        print(f"\n--- Tool calls ---\n{json.dumps(tool_calls, indent=2) if tool_calls else None}")

        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {})
                print(f"\n  Tool: {fn.get('name')}")
                print(f"  Args: {fn.get('arguments')}")
            print("\n[RESULT] Model successfully used native tool calling!")
        else:
            print("\n[RESULT] Model did NOT produce tool calls (returned text instead).")
            if content and ("```" in content):
                print("         It fell back to outputting code blocks in text.")

    except requests.exceptions.HTTPError as e:
        print(f"\n[ERROR] Server rejected tool-calling request: {e}")
        print(f"        Response: {e.response.text[:500]}")
    except Exception as e:
        print(f"\n[ERROR] {e}")


def main() -> None:
    print("Checking vLLM server...")
    if not wait_for_server(timeout=10):
        print("[ERROR] vLLM server not running on port 8000.")
        print("Start it with:")
        print(
            "  CUDA_VISIBLE_DEVICES=0,1 vllm serve "
            "/home/agoswami/_hackerreborn/aifsdk/_ckpts/omniagent-v1 "
            "--tensor-parallel-size 2 --port 8000 --served-model-name omniagent-v1"
        )
        sys.exit(1)

    test_text_based()
    test_tool_calling()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        "If TEST 2 produced valid tool_calls → model supports native tool calling.\n"
        "If TEST 2 returned text or errored → stick with litellm_textbased.\n"
        "\n"
        "NOTE: The chat_template in this model's tokenizer_config.json does NOT\n"
        "include tool-calling formatting, so native tool calling likely relies on\n"
        "vLLM's built-in tool parsing, which may or may not work well."
    )


if __name__ == "__main__":
    main()
