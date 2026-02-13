"""
Trial 05: Effective Batch Size — Measuring Dead Weight in GRPO

Shows how phitrain computes "train/effective_batch_size": the number of prompts
where the reward has non-zero variance across completions in the group.

If all completions for a prompt get the same reward (e.g., all 0.0), that prompt
produces zero advantage and zero gradient — it's dead weight.

Example scenarios:
  1. Reward bug (old code): django/sympy get "pytest not found" -> all rewards = 0
  2. Model too weak: fails all completions for a hard instance -> all rewards = 0
  3. Model too strong: solves all completions for an easy instance -> all rewards = 1.0

In all three cases, effective_batch_size excludes that prompt.

Reference: phitrain/phitrain/rl/rewards/reward_manager.py:365-370 (compute_metrics)
           phitrain/phitrain/rl/interaction_batch.py:322-339 (filter_prompts_by_reward_std)
"""

import torch

# ===========================================================================
# Scenario A: Reward bug affecting django/sympy (the bug we just fixed)
# batch_size=4, group_size=8
# ===========================================================================

print("=" * 70)
print("SCENARIO A: Reward bug (old code)")
print("62.5% of instances are django/sympy where pytest is not installed")
print("=" * 70)

scores_a = torch.tensor(
    [
        [0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0],  # scikit-learn: pytest works
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # django: "pytest not found" -> all 0
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # sympy: "pytest not found" -> all 0
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # matplotlib: pytest works
    ],
    dtype=torch.float32,
)

# Reference: reward_manager.py:365-366
reward_range_epsilon = 1e-6
score_range_a = scores_a.max(dim=1).values - scores_a.min(dim=1).values
nonzero_mask_a = score_range_a >= reward_range_epsilon

# Reference: reward_manager.py:369-370
effective_batch_size_a = int(nonzero_mask_a.sum().item())

print(f"Scores shape: {list(scores_a.shape)} (batch_size=4, group_size=8)")
print(f"Per-prompt score range: {score_range_a.tolist()}")
print(f"Non-zero reward mask:   {nonzero_mask_a.tolist()}")
print(f"train/effective_batch_size = {effective_batch_size_a}  (out of 4)")
print(f"-> {4 - effective_batch_size_a} prompts are dead weight\n")

# ===========================================================================
# Scenario B: After the reward fix
# Now django/sympy use correct test runners and produce real scores
# ===========================================================================

print("=" * 70)
print("SCENARIO B: After reward fix")
print("All repos use correct test commands")
print("=" * 70)

scores_b = torch.tensor(
    [
        [0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0],  # scikit-learn
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # django: runtests.py works!
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # sympy: bin/test works!
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # matplotlib
    ],
    dtype=torch.float32,
)

score_range_b = scores_b.max(dim=1).values - scores_b.min(dim=1).values
nonzero_mask_b = score_range_b >= reward_range_epsilon
effective_batch_size_b = int(nonzero_mask_b.sum().item())

print(f"Per-prompt score range: {score_range_b.tolist()}")
print(f"Non-zero reward mask:   {nonzero_mask_b.tolist()}")
print(f"train/effective_batch_size = {effective_batch_size_b}  (out of 4)")
print(f"-> {4 - effective_batch_size_b} prompts are dead weight\n")

# ===========================================================================
# Scenario C: Edge case — model too strong or too weak
# ===========================================================================

print("=" * 70)
print("SCENARIO C: Model too strong on some, too weak on others")
print("=" * 70)

scores_c = torch.tensor(
    [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # too easy: all correct -> no signal
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # too hard: all wrong -> no signal
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # just right: some pass, some fail
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # just right: one pass
    ],
    dtype=torch.float32,
)

score_range_c = scores_c.max(dim=1).values - scores_c.min(dim=1).values
nonzero_mask_c = score_range_c >= reward_range_epsilon
effective_batch_size_c = int(nonzero_mask_c.sum().item())

print(f"Per-prompt score range: {score_range_c.tolist()}")
print(f"Non-zero reward mask:   {nonzero_mask_c.tolist()}")
print(f"train/effective_batch_size = {effective_batch_size_c}  (out of 4)")
print()
print("Note: both 'too easy' (all 1.0) and 'too hard' (all 0.0) are dead weight.")
print("GRPO only learns from prompts where the group has VARIANCE in rewards.")
print()

# ===========================================================================
# Filtering: InteractionBatch.filter_prompts_by_reward_std
# Reference: interaction_batch.py:322-339
# ===========================================================================

print("=" * 70)
print("FILTERING: How filter_prompts_by_reward_std works")
print("=" * 70)

# Reference: interaction_batch.py:95-108
def compute_reward_std(scores: torch.Tensor) -> list[float]:
    """Mirrors InteractionBatch._compute_reward_std"""
    return scores.std(dim=1, unbiased=False).tolist()

min_std = 1e-4  # threshold used in CISPO

std_c = compute_reward_std(scores_c)
keep_indices = [i for i, s in enumerate(std_c) if s > min_std]

print(f"Using scenario C scores:")
print(f"Per-prompt std: {[f'{s:.6f}' for s in std_c]}")
print(f"Threshold (min_std): {min_std}")
print(f"Keep indices: {keep_indices}")
print(f"Filtered batch size: {len(keep_indices)} (from 4)")
print()

# Show the filtered scores
filtered_scores = scores_c[keep_indices]
print(f"Filtered scores (only prompts with learning signal):")
for i, idx in enumerate(keep_indices):
    print(f"  original prompt {idx}: {scores_c[idx].tolist()}")
print()

# ===========================================================================
# Impact on advantages after filtering
# ===========================================================================

print("=" * 70)
print("IMPACT: Advantages before vs after filtering")
print("=" * 70)

def calculate_advantages(scores: torch.Tensor) -> torch.Tensor:
    """Mirrors tuner_utils.calculate_advantages with prompt normalization"""
    adv = scores - scores.mean(dim=1, keepdim=True)
    adv = adv / (scores.std(dim=1, keepdim=True) + 1e-4)
    return adv

print("Before filtering (4 prompts):")
adv_before = calculate_advantages(scores_c)
for i in range(4):
    label = ["too easy", "too hard", "good", "good"][i]
    print(f"  prompt {i} ({label:>8}): advantages = {[f'{a:.4f}' for a in adv_before[i].tolist()]}")

print()
print("After filtering (2 prompts):")
adv_after = calculate_advantages(filtered_scores)
for i, idx in enumerate(keep_indices):
    print(f"  prompt {idx} (    good): advantages = {[f'{a:.4f}' for a in adv_after[i].tolist()]}")

print()
print("Key insight: filtering doesn't change the advantages for surviving prompts")
print("(since advantage normalization is per-prompt, not per-batch in GRPO).")
print("It only removes dead weight from the forward+backward pass.")
