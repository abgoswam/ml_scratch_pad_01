"""Reproduces the parser crash from job agoswami-p0-rlscaling-swe-job-tc-sbv-gnplj.

The bug: json.loads() can succeed but return a non-dict (e.g. a string),
which then crashes on .get() at line 68 of parsers.py.
"""

import json
import re

_TOOL_CALL_REGEX = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


# --- The three types of model output that hit the parser ---

# 1. Normal (works fine)
good_response = """
I'll list the files in the repo.
<tool_call>
{"name": "bash", "arguments": {"command": "ls -la"}}
</tool_call>
"""

# 2. Model emits a JSON string instead of a JSON object (CRASHES)
bad_response = """
Let me look at the code.
<tool_call>
"bash -c 'find . -name test_*.py'"
</tool_call>
"""

# 3. Invalid JSON (already handled gracefully)
invalid_json_response = """
<tool_call>
{name: bash, arguments: {command: ls}}
</tool_call>
"""


def parse_original(response_text: str) -> str:
    """Original parser logic (before fix)."""
    matches = _TOOL_CALL_REGEX.findall(response_text)
    if len(matches) != 1:
        return f"  Match count: {len(matches)} -> returns format error (handled)"

    raw = matches[0].strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        return f"  JSONDecodeError: {e} -> returns format error (handled)"

    # This is where it crashes:
    name = parsed.get("name", "")  # AttributeError if parsed is a str!
    return f"  Parsed successfully: name={name}"


# --- Run the examples ---

print("=" * 60)
print("ORIGINAL PARSER (before fix)")
print("=" * 60)

print("\n1. Good response (model produces correct JSON object):")
print(parse_original(good_response))

print("\n2. Invalid JSON response:")
print(parse_original(invalid_json_response))

print("\n3. Bad response (model produces a JSON *string*):")
print("   json.loads('\"hello\"') =", repr(json.loads('"hello"')), "  <- valid JSON, but it's a str!")
try:
    print(parse_original(bad_response))
except AttributeError as e:
    print(f"  CRASH: {e}")
    print(f"  This kills the entire asyncio.gather() batch of 128 agents!")


print()
print("=" * 60)
print("FIXED PARSER (after adding isinstance check)")
print("=" * 60)


def parse_fixed(response_text: str) -> str:
    """Fixed parser logic."""
    matches = _TOOL_CALL_REGEX.findall(response_text)
    if len(matches) != 1:
        return f"  Match count: {len(matches)} -> returns format error (handled)"

    raw = matches[0].strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        return f"  JSONDecodeError: {e} -> returns format error (handled)"

    # FIX: check type before calling .get()
    if not isinstance(parsed, dict):
        return f"  Got {type(parsed).__name__} instead of dict -> returns format error (handled)"

    name = parsed.get("name", "")
    return f"  Parsed successfully: name={name}"


print("\n3. Bad response (same input, now handled gracefully):")
print(parse_fixed(bad_response))
