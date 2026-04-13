# Local Single-Instance SWE-bench Debug Run

## Goal

Run 1 SWE-bench task for 1 turn locally to inspect:
1. The full system message (with baked-in XML tool definitions)
2. Raw vLLM response (JSON from `/v1/chat/completions`)
3. Parsed response (`Qwen35XMLParser` output)

## Prerequisites (already satisfied)

| Requirement | Status |
|---|---|
| GPU | 4x RTX A6000 (only need 1) |
| vllm | 0.11.2 installed |
| openai | 2.26.0 installed |
| transformers | 4.57.6 installed |
| datasets | 4.6.1 installed |
| Model | `/home/agoswami/_hackerreborn/aifsdk/_ckpts/Qwen3-14B-modified` |
| Docker | **Not needed** (not executing commands, just inspecting messages/output) |
| Harbor | **Not needed** |
| Ray | **Not needed** |

---

## Step 1: Start vLLM Server

In a **separate terminal**, run:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /home/agoswami/_hackerreborn/aifsdk/_ckpts/Qwen3-14B-modified \
  --served-model-name vllm_hosted_model \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --port 8000 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --disable-log-requests
```

Single GPU, no Ray, no data parallelism. Wait for the `Started server` log line before proceeding.

---

## Step 2: Run the Debug Script

```bash
cd /home/agoswami/_hackerreborn/aifsdk
python eval/swe_bench/debug_single_instance.py
```

Or with a specific task:

```bash
python eval/swe_bench/debug_single_instance.py --instance-id "django__django-16527"
```

---

## What the Script Does

### 2a. Load 1 SWE-bench Task

- Uses `datasets` library to load `princeton-nlp/SWE-bench_Verified` (test split)
- Picks the first task by default, or a specific `instance_id` via `--instance-id`

### 2b. Build Messages (using the real pipeline code)

- Imports `create_miniswe_message_formatter` from
  `phitrain/recipes/swe_agent/swe_agent/message_formatter.py`
- Loads the tokenizer from `_ckpts/Qwen3-14B-modified`
- Calls the formatter with the task's `problem_statement`
  - Internally this calls `get_augmented_system_content(tokenizer, system_template, tools=SWE_TOOL_SCHEMAS)`
  - Which runs `tokenizer.apply_chat_template(tools=...)` and extracts the system content
  - The result has XML tool definitions baked into the system message

**PRINTS**: full system message and user message

### 2c. Send Request to vLLM

- Uses `openai.OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")`
- Calls `client.chat.completions.create()` with:
  - `model="vllm_hosted_model"`
  - `messages=[system_msg, user_msg]`
  - `tools=[bash_tool]` (from `swe_agent/tools.py`)
  - `max_tokens=2048`, `temperature=0.6`

**PRINTS**: full raw response JSON (`ChatCompletion` object)

### 2d. Parse with Qwen35XMLParser

- Imports `Qwen35XMLParser` from `phitrain/recipes/common/parsers.py`
- Strips `<think>` blocks from the response content
- Calls `parser.parse_response()` on the stripped text
- Runs `check_format_warnings()` for format validation

**PRINTS**: parsed result — thinking, commands, errors, warnings

### 2e. Side-by-Side Comparison

Prints all three views together:

| View | Source | Shows |
|---|---|---|
| Raw content | `response.choices[0].message.content` | Full text with `<think>`, `<tool_call>` XML |
| vLLM tool_calls | `response.choices[0].message.tool_calls` | What vLLM's `qwen3_xml` parser extracted |
| Agent parsed | `Qwen35XMLParser.parse_response()` | What mini-swe-agent would extract (thinking, commands, errors) |

---

## Files

| File | Purpose |
|---|---|
| `eval/swe_bench/debug_single_instance.py` | The debug script (to be created) |
| `eval/swe_bench/LOCAL_DEBUG_PLAN.md` | This plan |

---

## Key Code References

| Component | File | Line |
|---|---|---|
| System/instance templates | `phitrain/recipes/swe_agent/configs/swebench.yaml` | 2-123 |
| Tool schema (bash_tool) | `phitrain/recipes/swe_agent/swe_agent/tools.py` | 9-27 |
| `get_augmented_system_content()` | `phitrain/recipes/swe_agent/swe_agent/tools.py` | 30-61 |
| Message formatter | `phitrain/recipes/swe_agent/swe_agent/message_formatter.py` | 22-60 |
| `Qwen35XMLParser` | `phitrain/recipes/common/parsers.py` | 245-306 |
| `check_format_warnings()` | `phitrain/recipes/common/parsers.py` | 41-145 |
| Chat template (modified) | `_ckpts/Qwen3-14B-modified/tokenizer_config.json` | (chat_template field) |
