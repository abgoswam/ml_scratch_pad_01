"""
Why validate_keys is only True for Mixformer models.

In async_vllm_worker.py line 175, weight key validation is gated:
    validate_keys = config.model_type == "mixformer"

The validation in yield_weights_for_rollout() does:
  1. Yields all weights from the HF training model (source).
  2. Remaps source keys via STACKED_PARAMS: q_proj -> qkv_proj, etc.
  3. Compares the remapped set against vLLM's expected_keys (from vLLM state_dict).
     If there's any mismatch => raises ValueError.

This script uses:
  - REAL HF source keys (from checkpoint index files)
  - REAL vLLM expected keys (from vLLM model classes instantiated on meta device)
No simulation.

Usage:
    conda run -n py312_0205_job_verify python trial_02_layer_name_mapping.py
"""

import json
from huggingface_hub import hf_hub_download


# ── Hardcoded stacked params mapping (copied from utils.py) ─────────────
STACKED_PARAMS = {
    "q_proj": "qkv_proj",
    "k_proj": "qkv_proj",
    "v_proj": "qkv_proj",
    "gate_proj": "gate_up_proj",
    "up_proj": "gate_up_proj",
}


def to_vllm_key(key: str) -> str:
    """Convert a HF weight key to the vLLM fused key (utils.py lines 87-92)."""
    for src, tgt in STACKED_PARAMS.items():
        if key.endswith(f".{src}.weight") or key.endswith(f".{src}.bias"):
            return key.replace(f".{src}.", f".{tgt}.")
    return key


def load_keys_from_index(path: str) -> list[str]:
    with open(path) as f:
        return sorted(json.load(f)["weight_map"].keys())


def load_hf_source_keys(ckpt_path: str = None, hf_repo: str = None) -> list[str]:
    """Load HF source keys from a local checkpoint or HF cache."""
    if ckpt_path:
        keys = load_keys_from_index(f"{ckpt_path}/model.safetensors.index.json")
    else:
        index_path = hf_hub_download(hf_repo, "model.safetensors.index.json", local_files_only=True)
        keys = load_keys_from_index(index_path)

    return [
        k for k in keys
        if "layers.0." in k or "embed" in k or "lm_head" in k or k == "model.norm.weight"
    ]


def get_real_vllm_keys(model_id: str, vllm_cls_path: str) -> list[str]:
    """Instantiate vLLM model class on meta device to get real expected keys.

    Uses lazy imports to avoid initializing distributed state unless needed.
    vllm_cls_path: e.g. "vllm.model_executor.models.llama.LlamaForCausalLM"
    """
    import os
    os.environ.setdefault('VLLM_LOGGING_LEVEL', 'ERROR')
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29500')

    import torch
    import importlib
    from vllm.config import VllmConfig, ModelConfig, CompilationConfig
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    # Initialize distributed state (idempotent check)
    if not torch.distributed.is_initialized():
        init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
        initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    # Import model class
    module_path, cls_name = vllm_cls_path.rsplit(".", 1)
    vllm_cls = getattr(importlib.import_module(module_path), cls_name)

    mc = ModelConfig(model=model_id, task="generate", tokenizer=model_id,
                     dtype="bfloat16", trust_remote_code=True, max_model_len=128)
    cc = CompilationConfig(level=0)
    vc = VllmConfig(model_config=mc, compilation_config=cc)

    with torch.device("meta"):
        model = vllm_cls(vllm_config=vc)

    all_keys = sorted(model.state_dict().keys())
    return [
        k for k in all_keys
        if "layers.0." in k or "embed" in k or "lm_head" in k or k == "model.norm.weight"
    ]


def run_validation(
    name: str,
    hf_source_keys: list[str],
    vllm_expected_keys: list[str],
    validate_keys: bool,
):
    """Run the exact validation logic from utils.py lines 83-103."""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  validate_keys={validate_keys}")
    print(f"{'='*70}")

    # Show source -> remapped
    print(f"\n  HF source key  ->  remapped (via STACKED_PARAMS):")
    for k in hf_source_keys:
        v = to_vllm_key(k)
        if k != v:
            print(f"    {k}")
            print(f"      -> {v}")
        else:
            print(f"    {k}  (unchanged)")

    # utils.py line 83-84: early return if not validating
    if not validate_keys:
        print(f"\n  validate_keys=False => skip comparison (production behavior).")
        return

    # utils.py lines 94-103: the actual comparison
    expected_from_yielded = {to_vllm_key(k) for k in hf_source_keys}
    expected_keys = set(vllm_expected_keys)

    missing_target = expected_keys - expected_from_yielded
    extra_yielded = expected_from_yielded - expected_keys

    print(f"\n  Comparison (utils.py lines 94-103):")
    print(f"    remapped source keys:  {sorted(expected_from_yielded)}")
    print(f"    vLLM expected keys:    {sorted(expected_keys)}")

    if missing_target or extra_yielded:
        print(f"\n  ** RAISES ValueError **")
        if missing_target:
            print(f"    Missing (vLLM expects, source didn't produce):")
            for k in sorted(missing_target):
                print(f"      - {k}")
        if extra_yielded:
            print(f"    Extra (source produced, vLLM doesn't expect):")
            for k in sorted(extra_yielded):
                print(f"      - {k}")
    else:
        print(f"\n  VALIDATION PASSES")


# ── Load real keys ──────────────────────────────────────────────────────

LLAMA_CKPT = "/home/agoswami/_hackerreborn/aifsdk/_ckpts/llama"
QWEN_REPO = "Qwen/Qwen3-4B-Instruct-2507"

# HF source keys (from checkpoint index files)
llama_hf_keys = load_hf_source_keys(ckpt_path=LLAMA_CKPT)
qwen_hf_keys = load_hf_source_keys(hf_repo=QWEN_REPO)

# Real vLLM expected keys (from vLLM model classes on meta device)
llama_vllm_keys = get_real_vllm_keys(LLAMA_CKPT, "vllm.model_executor.models.llama.LlamaForCausalLM")
qwen_vllm_keys = get_real_vllm_keys(QWEN_REPO, "vllm.model_executor.models.qwen3.Qwen3ForCausalLM")

# ── Run ─────────────────────────────────────────────────────────────────

print("STACKED_PARAMS (hardcoded in utils.py):")
for src, tgt in STACKED_PARAMS.items():
    print(f"  {src:15s} -> {tgt}")

# Case 1: Llama with validate_keys=True — PASSES
run_validation("Llama (validate_keys=True)", llama_hf_keys, llama_vllm_keys, validate_keys=True)

# Case 2: Qwen3 with validate_keys=False — skips (production behavior)
run_validation("Qwen3 (validate_keys=False, production)", qwen_hf_keys, qwen_vllm_keys, validate_keys=False)

# Case 3: Qwen3 with validate_keys=True — FAILS
run_validation("Qwen3 (validate_keys=True, what-if)", qwen_hf_keys, qwen_vllm_keys, validate_keys=True)
