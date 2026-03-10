# Proposal: Add `enable_thinking` Support to omniagent-v1 Chat Template

## Problem

omniagent-v1 always produces `<think>` blocks in its output. This behavior is baked into the model weights via SFT — the model emits `<think>` regardless of what system prompt is provided. There is currently no way to suppress thinking at inference time.

Qwen3-14B (a comparable thinking model) solves this with an `enable_thinking` parameter in its chat template, giving callers control over whether the model thinks or not.

## Investigation

We tested three configurations across both models using the vLLM `/v1/chat/completions` API with a custom system prompt (the swebench eval prompt — not the hardcoded thinking prompt).

### omniagent-v1 (original template, no `enable_thinking` support)

| Test | `<think>` in output? | Response length |
|---|---|---|
| Baseline (no param) | Yes | 3,792 chars |
| `enable_thinking=True` | Yes | 7,651 chars |
| `enable_thinking=False` | Yes | 7,651 chars |

The `enable_thinking` parameter is ignored because the template doesn't handle it.

### Qwen3-14B (native `enable_thinking` support)

| Test | `<think>` in output? | Response length |
|---|---|---|
| Baseline (no param) | Yes | 1,620 chars |
| `enable_thinking=True` | Yes | 2,103 chars |
| `enable_thinking=False` | **No** | **306 chars** |

When `enable_thinking=False` is passed via `chat_template_kwargs`, the template prepends an empty `<think>\n\n</think>\n\n` block to the assistant turn. This signals to the model that thinking is complete, and it skips straight to the answer.

## Proposed Change

A one-line change to `tokenizer_config.json` in the omniagent-v1 checkpoint. The end of the chat template changes from:

```jinja
{% if add_generation_prompt %}{{ '<|im_start|>assistant<|im_sep|>' }}{% endif %}
```

to:

```jinja
{% if add_generation_prompt %}{{ '<|im_start|>assistant<|im_sep|>' }}{% if enable_thinking is defined and enable_thinking is false %}{{ '<think>\n\n</think>\n\n' }}{% endif %}{% endif %}
```

This is the same pattern used by Qwen3-14B.

## Verification

After applying the change, we re-ran the same tests on omniagent-v1:

| Test | `<think>` in output? | Response length |
|---|---|---|
| Baseline (no param) | Yes | 3,792 chars |
| `enable_thinking=True` | Yes | 7,651 chars |
| `enable_thinking=False` | **No** | **61 chars** |

The change works as intended:
- **Default behavior is unchanged** — the model still thinks when no param is passed or when `enable_thinking=True`.
- **`enable_thinking=False` suppresses thinking** — the model goes straight to the answer.

## Usage

Pass `enable_thinking=False` via `chat_template_kwargs` in the vLLM API request:

```json
{
  "model": "omniagent-v1",
  "messages": [...],
  "chat_template_kwargs": {"enable_thinking": false}
}
```

Note: the top-level `enable_thinking` parameter (without `chat_template_kwargs`) does not work — vLLM does not pass top-level params to the Jinja template.

## Impact

- Enables toggling thinking on/off for omniagent-v1 at inference time
- No model retraining required — template-only change
- Backward compatible — existing callers that don't pass the param see no change
- Aligns omniagent-v1's API surface with Qwen3-14B
