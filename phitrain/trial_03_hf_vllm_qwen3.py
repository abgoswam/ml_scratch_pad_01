"""
Compare logits between HF and vLLM for Qwen3-4B-Instruct.

Shows that both produce identical logits, confirming weight tying
(lm_head = embed_tokens) works the same on both sides.

Usage:
    conda run -n py312_0205_job_verify python trial_03_hf_vllm_qwen3.py
"""

import os
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
PROMPT = "tell me a fun fact:"
TOP_K = 20
DTYPE = "float32"  # "bfloat16" or "float32"


def get_hf_logits() -> tuple[torch.Tensor, list[int]]:
    """Load HF model on GPU:0, return full logits at last position."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading HF model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    torch_dtype = getattr(torch, DTYPE)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, local_files_only=True, dtype=torch_dtype, device_map="cuda:0",
    )

    # Confirm weight tying
    embed_w = model.model.embed_tokens.weight
    lm_head_w = model.lm_head.weight
    print(f"  embed_tokens.weight data_ptr: {embed_w.data_ptr()}")
    print(f"  lm_head.weight      data_ptr: {lm_head_w.data_ptr()}")
    print(f"  Same tensor? {embed_w.data_ptr() == lm_head_w.data_ptr()}")

    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to("cuda:0")
    print(f"  Input tokens: {input_ids[0].tolist()}")

    with torch.no_grad():
        output = model(input_ids)
    logits = output.logits[0, -1, :].float().cpu()  # last position, float32 for comparison

    del model
    torch.cuda.empty_cache()
    return logits, input_ids[0].tolist()


def get_vllm_logprobs() -> dict[int, float]:
    """Load vLLM model on GPU:1, return top logprobs at first generated token."""
    from vllm import LLM, SamplingParams

    print("Loading vLLM model...", flush=True)
    llm = LLM(
        model=MODEL_ID,
        enforce_eager=True,
        max_model_len=128,
        gpu_memory_utilization=0.5,
        tensor_parallel_size=1,
        dtype=DTYPE,
    )

    params = SamplingParams(
        max_tokens=1,
        temperature=0,
        logprobs=TOP_K,
    )
    outputs = llm.generate([PROMPT], params)
    token_logprobs = outputs[0].outputs[0].logprobs[0]  # first (only) generated token

    # token_id -> logprob
    result = {token_id: info.logprob for token_id, info in token_logprobs.items()}

    del llm
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    # Step 1: HF logits
    hf_logits, token_ids = get_hf_logits()
    hf_log_probs = torch.log_softmax(hf_logits, dim=-1)

    # Top-K from HF
    hf_topk_vals, hf_topk_ids = torch.topk(hf_log_probs, TOP_K)

    print(f"\n{'='*70}")
    print(f"  HF top-{TOP_K} predictions after \"{PROMPT}\"")
    print(f"{'='*70}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    for i in range(TOP_K):
        tid = hf_topk_ids[i].item()
        lp = hf_topk_vals[i].item()
        print(f"    {i+1:2d}. token={tid:6d}  logprob={lp:8.4f}  text={tokenizer.decode([tid])!r}")

    # Step 2: vLLM logprobs
    vllm_logprobs = get_vllm_logprobs()

    print(f"\n{'='*70}")
    print(f"  vLLM top-{TOP_K} predictions after \"{PROMPT}\"")
    print(f"{'='*70}")
    vllm_sorted = sorted(vllm_logprobs.items(), key=lambda x: -x[1])
    for i, (tid, lp) in enumerate(vllm_sorted[:TOP_K]):
        print(f"    {i+1:2d}. token={tid:6d}  logprob={lp:8.4f}  text={tokenizer.decode([tid])!r}")

    # Step 3: Compare
    print(f"\n{'='*70}")
    print(f"  Comparison (top-{TOP_K} tokens)")
    print(f"{'='*70}")

    hf_top_ids = [hf_topk_ids[i].item() for i in range(TOP_K)]
    vllm_top_ids = [tid for tid, _ in vllm_sorted[:TOP_K]]

    matching_ids = set(hf_top_ids) & set(vllm_top_ids)
    print(f"  Top-{TOP_K} token overlap: {len(matching_ids)}/{TOP_K}")

    # Compare logprobs for overlapping tokens
    diffs = []
    print(f"\n  {'token':>8s}  {'HF logprob':>12s}  {'vLLM logprob':>12s}  {'diff':>10s}  text")
    print(f"  {'-'*60}")
    for tid in hf_top_ids:
        if tid in vllm_logprobs:
            hf_lp = hf_log_probs[tid].item()
            vllm_lp = vllm_logprobs[tid]
            diff = abs(hf_lp - vllm_lp)
            diffs.append(diff)
            print(f"  {tid:8d}  {hf_lp:12.6f}  {vllm_lp:12.6f}  {diff:10.6f}  {tokenizer.decode([tid])!r}")

    if diffs:
        print(f"\n  Max abs diff:  {max(diffs):.6f}")
        print(f"  Mean abs diff: {np.mean(diffs):.6f}")

        threshold = 0.1 if DTYPE == "bfloat16" else 0.01
        if max(diffs) < threshold:
            print(f"\n  LOGITS MATCH (within {DTYPE} precision)")
        else:
            print(f"\n  LOGITS DIFFER beyond {DTYPE} precision (threshold={threshold})")
