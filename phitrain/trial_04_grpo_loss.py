"""
Trial 04: GRPO Loss Calculation — Step by Step

Traces the exact GRPO loss computation from phitrain, using a minimal example:
  - 2 prompts, group_size=2
  - Prompt 0: rewards [1.0, 0.0] (has variance — one completion correct, one wrong)
  - Prompt 1: rewards [0.0, 0.0] (all zero — model failed on both completions)

Key takeaway: prompt 1 produces ZERO loss and ZERO gradient, but still runs through
the full forward+backward pass. Its tokens are dead weight.

Reference: phitrain/phitrain/rl/tuners/grpo/grpo_worker.py (compute_loss)
           phitrain/phitrain/rl/tuners/tuner_utils.py (calculate_advantages)
"""

import torch

# ===========================================================================
# Step 1: Rewards → Advantages
# Reference: tuner_utils.py:26-34
# ===========================================================================

print("=" * 70)
print("STEP 1: Rewards -> Advantages")
print("=" * 70)

# rewards: [num_prompts, group_size]
rewards = torch.tensor(
    [
        [1.0, 0.0],  # prompt 0: correct fix vs wrong fix
        [0.0, 0.0],  # prompt 1: both completions failed
    ],
    dtype=torch.float32,
)
print(f"Rewards:\n{rewards}\n")

# Line 31: subtract per-row mean
advantages = rewards - rewards.mean(dim=1, keepdim=True)
print(f"After subtracting row mean:")
print(f"  prompt 0 mean = {rewards[0].mean().item():.1f} -> advantages = {advantages[0].tolist()}")
print(f"  prompt 1 mean = {rewards[1].mean().item():.1f} -> advantages = {advantages[1].tolist()}")
print()

# Line 33-34: prompt-level normalization (GRPO style)
std = rewards.std(dim=1, keepdim=True)
advantages = advantages / (std + 1e-4)
print(f"After dividing by (std + 1e-4):")
print(f"  prompt 0 std = {std[0].item():.4f} -> advantages = {advantages[0].tolist()}")
print(f"  prompt 1 std = {std[1].item():.4f} -> advantages = {advantages[1].tolist()}")
print()

# ===========================================================================
# Step 2: Simulate packing — replicate advantage per token
# Reference: interaction_batch.py pack() and packing.py PackedBatch.from_indexed_sequences
# ===========================================================================

print("=" * 70)
print("STEP 2: Pack advantages per token")
print("=" * 70)

# Each completion has 4 tokens (prompt + response), last 2 are response tokens.
# In practice sequences are much longer; keeping it small for clarity.
seq_len = 4
num_response_tokens = 2

# Flatten: [p0c0, p0c1, p1c0, p1c1]
advantages_flat = advantages.flatten()  # [4 values]
print(f"Flat advantages (per completion): {advantages_flat.tolist()}")

# Replicate per token: each token in a completion gets that completion's advantage
# Shape: [4 completions, seq_len]
token_advantages = advantages_flat.unsqueeze(1).expand(-1, seq_len)
print(f"Token advantages (each row = one completion's tokens):\n{token_advantages}\n")

# Assistant mask: only response tokens get loss (last 2 of each sequence)
assistant_mask = torch.zeros(4, seq_len, dtype=torch.bool)
assistant_mask[:, -num_response_tokens:] = True
print(f"Assistant mask:\n{assistant_mask.int()}\n")

# ===========================================================================
# Step 3: Compute GRPO loss (PPO-clip style)
# Reference: grpo_worker.py:86-98
# ===========================================================================

print("=" * 70)
print("STEP 3: Compute GRPO loss")
print("=" * 70)

epsilon_low = 0.2
epsilon_high = 0.2

# Simulate log-probs. On first update, policy hasn't changed so ratio ~ 1.0.
# We add small perturbations to make it realistic.
torch.manual_seed(42)
per_token_logps = torch.randn(4, seq_len) * 0.1 - 2.0  # current policy
old_per_token_logps = per_token_logps.detach() + torch.randn(4, seq_len) * 0.01  # old policy (close)

# Lines 90-92: shift mask and advantages (loss is computed for next-token prediction)
shifted_mask = assistant_mask[:, 1:]
shifted_advantages = token_advantages[:, 1:]

# Line 94: importance sampling ratio
coef_1 = torch.exp(per_token_logps[:, :-1] - old_per_token_logps[:, :-1])
print(f"Importance ratios (coef_1) — should be ~1.0:")
print(f"  prompt 0, comp 0: {coef_1[0].tolist()}")
print(f"  prompt 1, comp 0: {coef_1[2].tolist()}")
print()

# Line 95: clipped ratio
coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)

# Lines 96-98: PPO-clip objective
per_token_loss1 = coef_1 * shifted_advantages
per_token_loss2 = coef_2 * shifted_advantages
per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

print("Per-token loss (before masking):")
for i in range(4):
    prompt_idx = i // 2
    comp_idx = i % 2
    print(f"  prompt {prompt_idx}, comp {comp_idx}: {per_token_loss[i].tolist()}")
print()

# ===========================================================================
# Step 4: Aggregate loss
# Reference: grpo_worker.py:114-118 (sequence normalization)
# ===========================================================================

print("=" * 70)
print("STEP 4: Aggregate loss")
print("=" * 70)

# With sequence normalization: each token's loss is divided by its sequence's valid token count
# Reference: grpo_worker.py:35-40
valid_seqlens = shifted_mask.sum(dim=1, keepdim=True).float() + 1e-3  # per sequence
loss_scale = valid_seqlens.expand_as(shifted_mask)

masked_loss = (per_token_loss * shifted_mask) / loss_scale
print(f"Masked + scaled loss per completion:")
for i in range(4):
    prompt_idx = i // 2
    comp_idx = i % 2
    comp_loss = masked_loss[i].sum().item()
    print(f"  prompt {prompt_idx}, comp {comp_idx}: {comp_loss:.6f}")
print()

raw_loss = masked_loss.sum()
print(f"Raw loss (sum over all tokens): {raw_loss.item():.6f}")

# Line 239-241: scale by dp_size / (train_batch_size * group_size)
train_batch_size = 2
group_size = 2
dp_size = 1
scale = dp_size / (train_batch_size * group_size)
final_loss = raw_loss * scale
print(f"Scale factor: dp_size / (batch_size * group_size) = {dp_size} / ({train_batch_size} * {group_size}) = {scale}")
print(f"Final loss: {final_loss.item():.6f}")
print()

# ===========================================================================
# Step 5: Show the dead weight
# ===========================================================================

print("=" * 70)
print("STEP 5: Dead weight analysis")
print("=" * 70)

prompt_0_loss = masked_loss[:2].sum().item()
prompt_1_loss = masked_loss[2:].sum().item()
print(f"Loss from prompt 0 (has variance): {prompt_0_loss:.6f}")
print(f"Loss from prompt 1 (all zero):     {prompt_1_loss:.6f}")
print()
print(f"Prompt 1 contributes {prompt_1_loss:.1f} loss and {prompt_1_loss:.1f} gradient,")
print(f"but its {2 * num_response_tokens} response tokens still run through forward + backward.")
print(f"With the reward bug affecting ~62.5% of instances, most of the batch was dead weight.")
