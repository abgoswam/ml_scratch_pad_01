# Qwen3-4B THOROUGH per-GPU memory map during actor forward pass
# ================================================================
# Setup: FSDP across 4 GPUs, manual_offload=True, micro_batch=[1, 6144]

GiB = 1024**3
MiB = 1024**2

# Model arch
hidden = 2560
ff = 9728
n_layers = 36
n_heads = 32
n_kv_heads = 8
head_dim = 128
qkv_dim = n_heads * head_dim  # 4096
vocab = 151936
total_params = 4_411_599_360
n_gpus = 4
seq = 6144
B = 1
bf16 = 2
fp32 = 4

print('=' * 75)
print('PER-GPU MEMORY MAP: Qwen3-4B, FSDP/4, micro_batch=[1, 6144]')
print('=' * 75)

# =====================================================================
# A. STATIC ALLOCATIONS (present throughout on_gpu() context)
# =====================================================================
print('\n--- A. STATIC ALLOCATIONS (entire on_gpu() scope) ---')

# A1. FSDP-sharded model params (bf16)
a1 = (total_params * bf16) / n_gpus
print(f'  A1. Model params (1/{n_gpus} FSDP shard, bf16):     {a1/GiB:8.2f} GiB')

# A2. Optimizer states (AdamW: exp_avg + exp_avg_sq, fp32, sharded)
a2_avg = (total_params * fp32) / n_gpus
a2_sq = (total_params * fp32) / n_gpus
a2 = a2_avg + a2_sq
print(f'  A2. Optimizer exp_avg (fp32, sharded):              {a2_avg/GiB:8.2f} GiB')
print(f'      Optimizer exp_avg_sq (fp32, sharded):           {a2_sq/GiB:8.2f} GiB')
print(f'      Optimizer total:                                {a2/GiB:8.2f} GiB')

# A3. CUDA context + Ray/misc overhead
a3 = 0.8 * GiB  # typically 0.5-1 GiB
print(f'  A3. CUDA context + Ray overhead:                    {a3/GiB:8.2f} GiB (est.)')

# A4. Other processes on GPU 0 (from error msg)
a4 = 524 * MiB  # 2 processes x 262 MiB
print(f'  A4. Other processes on GPU 0:                       {a4/GiB:8.2f} GiB (from error)')

static = a1 + a2 + a3 + a4
print(f'  --- Static subtotal:                                {static/GiB:8.2f} GiB')

# =====================================================================
# B. FSDP ALL-GATHER (temporary per layer during forward)
# =====================================================================
print('\n--- B. FSDP ALL-GATHER (per-layer, temporary) ---')
params_per_layer = 100_935_680
# FSDP all-gathers the FULL layer params temporarily on each GPU
b1 = params_per_layer * bf16
# Last layer has reshard_after_forward=False, so stays full
print(f'  B1. One layer all-gather (bf16):                    {b1/GiB:8.4f} GiB ({b1/MiB:.0f} MiB)')

# =====================================================================
# C. ACTIVATIONS STORED FOR BACKWARD (per transformer layer)
# =====================================================================
print('\n--- C. ACTIVATIONS (stored for backward, per layer) ---')

# Trace through Qwen3DecoderLayer forward:
# 1. input_layernorm(hidden_states) -- RMSNorm saves input
c_norm_input = B * seq * hidden * bf16
print(f'  C1. input_layernorm input:                            {c_norm_input/MiB:7.1f} MiB')

# 2. self_attn:
#    q = q_norm(q_proj(x).view()) -- q_proj saves input (shared), q_norm saves its input
#    q_proj output = [B, seq, qkv_dim=4096]
c_qproj_out = B * seq * qkv_dim * bf16
print(f'  C2. q_proj output (saved by q_norm bwd):             {c_qproj_out/MiB:7.1f} MiB')

#    k_proj output = [B, seq, n_kv_heads*head_dim=1024]
c_kproj_out = B * seq * (n_kv_heads * head_dim) * bf16
print(f'  C3. k_proj output (saved by k_norm bwd):             {c_kproj_out/MiB:7.1f} MiB')

#    v_proj output = [B, seq, n_kv_heads*head_dim=1024]
c_vproj_out = B * seq * (n_kv_heads * head_dim) * bf16
print(f'  C4. v_proj output:                                   {c_vproj_out/MiB:7.1f} MiB')

#    Flash attention saves: O (output) + logsumexp
c_flash_out = B * seq * qkv_dim * bf16  # [B, seq, n_heads * head_dim]
c_flash_lse = B * n_heads * seq * fp32   # [B, n_heads, seq]
print(f'  C5. Flash attn output:                               {c_flash_out/MiB:7.1f} MiB')
print(f'  C6. Flash attn logsumexp:                            {c_flash_lse/MiB:7.2f} MiB')

#    o_proj saves input [B, seq, qkv_dim] -- same memory as flash output (view)
#    So NOT double counted

# 3. post_attention_layernorm: saves input (residual + attn_output)
c_post_norm = B * seq * hidden * bf16
print(f'  C7. post_attn_layernorm input:                       {c_post_norm/MiB:7.1f} MiB')

# 4. MLP:
#    gate = gate_proj(x)  -- gate_proj saves input (shared with up_proj, = norm output)
#    Note: Linear saves input, but norm output is NOT the same as norm input
#    norm output ~ gate_proj input: [B, seq, hidden]
c_mlp_input = B * seq * hidden * bf16
print(f'  C8. MLP input (gate/up_proj input):                  {c_mlp_input/MiB:7.1f} MiB')

#    silu(gate) -- SiLU saves its input (gate_proj output)
c_gate_out = B * seq * ff * bf16
print(f'  C9. gate_proj output (saved by SiLU bwd):            {c_gate_out/MiB:7.1f} MiB')

#    silu(gate) * up -- element-wise mul saves BOTH inputs
#    silu_result: [B, seq, ff]  (new tensor)
c_silu_result = B * seq * ff * bf16
print(f'  C10. silu(gate) result (saved by mul bwd):           {c_silu_result/MiB:7.1f} MiB')
#    up_proj output: [B, seq, ff]
c_up_out = B * seq * ff * bf16
print(f'  C11. up_proj output (saved by mul bwd):              {c_up_out/MiB:7.1f} MiB')

#    down_proj saves input (= silu(gate) * up): [B, seq, ff]
c_down_input = B * seq * ff * bf16
print(f'  C12. down_proj input (= gate*up result):             {c_down_input/MiB:7.1f} MiB')

per_layer_act = (c_norm_input + c_qproj_out + c_kproj_out + c_vproj_out +
                 c_flash_out + c_flash_lse + c_post_norm + c_mlp_input +
                 c_gate_out + c_silu_result + c_up_out + c_down_input)
print(f'  --- Per layer total:                                 {per_layer_act/MiB:7.1f} MiB ({per_layer_act/GiB:.3f} GiB)')

all_layers_act = per_layer_act * n_layers
print(f'  --- All {n_layers} layers:                                  {all_layers_act/GiB:8.2f} GiB')

# =====================================================================
# D. POST-TRANSFORMER ACTIVATIONS
# =====================================================================
print('\n--- D. POST-TRANSFORMER ACTIVATIONS ---')

# Final RMSNorm input
d_final_norm = B * seq * hidden * bf16
print(f'  D1. Final RMSNorm input:                             {d_final_norm/MiB:7.1f} MiB')

# LM head (Linear) saves input: [B, seq, hidden]
d_lm_input = B * seq * hidden * bf16
print(f'  D2. LM head input:                                   {d_lm_input/MiB:7.1f} MiB')

# Logits tensor: [B, seq, vocab] in bf16
d_logits = B * seq * vocab * bf16
print(f'  D3. Logits [1, {seq}, {vocab}]:                {d_logits/GiB:8.2f} GiB ({d_logits/MiB:.0f} MiB)')

# logprobs_from_logits with flash cross_entropy:
# flat_logits is a view (no extra mem)
# cross_entropy_loss output: [seq] -- tiny
# Without flash CE: F.log_softmax creates full [B, seq, vocab] copy!
d_ce_fused = B * seq * fp32  # tiny output
d_ce_naive = B * (seq - 1) * vocab * bf16  # full log_softmax output
print(f'  D4a. Cross entropy (fused, flash_attn):              {d_ce_fused/MiB:7.3f} MiB')
print(f'  D4b. Cross entropy (naive log_softmax):              {d_ce_naive/GiB:8.2f} GiB  *** IF NO FLASH_ATTN ***')

post_act = d_final_norm + d_lm_input + d_logits + d_ce_fused
print(f'  --- Post-transformer total (w/ fused CE):            {post_act/GiB:8.2f} GiB')

# =====================================================================
# E. AUTOGRAD GRAPH OVERHEAD
# =====================================================================
print('\n--- E. AUTOGRAD GRAPH OVERHEAD ---')
# Each saved tensor has a TensorNode (~100-200 bytes)
# Each operation has a FunctionNode (~200-500 bytes)
# Rough estimate: ~10 ops per layer x 36 layers x ~1KB = negligible
# But PyTorch also keeps references, metadata, etc.
e1 = 0.3 * GiB  # conservative estimate for graph metadata
print(f'  E1. Autograd graph metadata:                        {e1/GiB:8.2f} GiB (est.)')

# =====================================================================
# F. FSDP COMMUNICATION BUFFERS
# =====================================================================
print('\n--- F. FSDP BUFFERS ---')
# All-gather output buffer (reused), reduce-scatter buffer
# Typically 1-2 layer sizes
f1 = 2 * params_per_layer * bf16
print(f'  F1. FSDP comm buffers (~2 layers):                  {f1/GiB:8.2f} GiB ({f1/MiB:.0f} MiB)')

# =====================================================================
# GRAND TOTAL
# =====================================================================
print('\n' + '=' * 75)
print('GRAND TOTAL (per GPU, peak during forward pass)')
print('=' * 75)
total_act = all_layers_act + post_act
items = [
    ('A. Model params (FSDP shard, bf16)',      a1),
    ('A. Optimizer states (AdamW, fp32)',        a2),
    ('A. CUDA context + Ray',                   a3),
    ('A. Other processes on GPU 0',             a4),
    ('B. FSDP all-gather (1 layer)',            b1),
    ('C. Layer activations (36 layers)',        all_layers_act),
    ('D. Post-transformer (logits, etc.)',      post_act),
    ('E. Autograd graph overhead',              e1),
    ('F. FSDP comm buffers',                    f1),
]
cumulative = 0
for name, val in items:
    cumulative += val
    print(f'  {name:45s}: {val/GiB:6.2f} GiB  (cum: {cumulative/GiB:5.2f} GiB)')

print(f'  {"":45s}  -----------')
print(f'  {"ESTIMATED TOTAL":45s}: {cumulative/GiB:6.2f} GiB')
print(f'  {"GPU CAPACITY":45s}: {44.42:6.2f} GiB')
print(f'  {"ACTUAL USAGE (from error)":45s}: {43.85:6.2f} GiB')
print(f'  {"ESTIMATED DEFICIT":45s}: {cumulative/GiB - 44.42:+6.2f} GiB')

print(f'\n--- WHAT IF we remove optimizer from GPU during fwd/bwd? ---')
no_opt = cumulative - a2
print(f'  Total without optimizer:                            {no_opt/GiB:6.2f} GiB')
print(f'  Headroom:                                           {44.42 - no_opt/GiB:6.2f} GiB')

# Also check: what if logprobs uses naive log_softmax (no flash_attn CE)?
print(f'\n--- WHAT IF flash_attn cross_entropy is NOT available? ---')
naive_extra = d_ce_naive - d_ce_fused
print(f'  Additional cost of naive log_softmax:               {naive_extra/GiB:6.2f} GiB')
print(f'  Total with naive CE:                                {(cumulative + naive_extra)/GiB:6.2f} GiB')
