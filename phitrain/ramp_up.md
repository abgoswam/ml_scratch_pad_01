# 3-Day Learning Plan for Phitrain/GRPO (with Library File References)

---

## **Day 1: Foundations (PyTorch Distributed + Transformers)**

### Morning: **PyTorch Distributed Basics**

| Topic | Resource | Used In Library |
|-------|----------|-----------------|
| `torch.distributed` basics | [PyTorch Distributed Tutorial](https://pytorch.org/tutorials/intermediate/dist_tuto.html) | [ray_utils.py#L93](phitrain/rl/ray_utils.py) - `maybe_init_process_group()` |
| FSDP (`fully_shard`) | [FSDP Getting Started](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) | [data_parallel.py](phitrain/models/parallellisms/data_parallel.py) - `apply_fsdp()` |
| Device Mesh | [DeviceMesh docs](https://pytorch.org/docs/stable/distributed.tensor.parallel.html) | [distributed_layout.py#L100](phitrain/rl/distributed_layout.py) - `init_device_mesh()` |
| Tensor Parallelism | PyTorch TP docs | [tensor_parallel.py](phitrain/models/parallellisms/tensor_parallel.py) - `apply_tp()` |
| Activation Checkpointing | PyTorch AC docs | [activation_checkpoint.py](phitrain/models/parallellisms/activation_checkpoint.py) - `apply_ac()` |

**Key concepts to understand**:
- What is "sharding" and why FSDP shards model parameters across GPUs
- Difference between Data Parallel (DP), Tensor Parallel (TP), and Pipeline Parallel
- How `DeviceMesh` organizes GPUs into a logical grid (e.g., 2 nodes × 4 GPUs)
- Why `reshard_after_forward=True` saves memory but costs communication
- Mixed precision: why train in fp32 but compute in bf16

**Practice**:
```python
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard

# 1. Initialize process group (run with torchrun --nproc_per_node=2)
dist.init_process_group("nccl")
rank = dist.get_rank()

# 2. Create a simple model and apply FSDP
model = torch.nn.Linear(1024, 1024).cuda()
mesh = init_device_mesh("cuda", (dist.get_world_size(),))
fully_shard(model, mesh=mesh)

# 3. Verify parameters are sharded
print(f"Rank {rank}: param shape = {model.weight.shape}")  # Should be smaller than 1024x1024

# 4. Run a forward pass and check gradients sync
x = torch.randn(32, 1024).cuda()
loss = model(x).sum()
loss.backward()
print(f"Rank {rank}: grad norm = {model.weight.grad.norm()}")
```

**Key file to study**: [actor.py#L20-L25](phitrain/rl/models/actor.py) - See how all parallelism strategies are imported and applied together

---

### Afternoon: **Hugging Face Ecosystem**

| Topic | Resource | Used In Library |
|-------|----------|-----------------|
| `AutoTokenizer` | [HF Tokenizers Guide](https://huggingface.co/docs/transformers/main_classes/tokenizer) | [tutorial_grpo.py#L52](scripts/rl/tutorial_grpo.py) - `AutoTokenizer.from_pretrained()` |
| `apply_chat_template()` | HF Chat Templates | [chat_dataset.py#L87-L90](phitrain/datasets/rl/chat/chat_dataset.py) - tokenizing messages |
| `PreTrainedModel` loading | [HF Quick Tour](https://huggingface.co/docs/transformers/quicktour) | [actor.py#L95-L120](phitrain/rl/models/actor.py) - `configure_model()` |
| `datasets.load_dataset()` | [HF Datasets Tutorial](https://huggingface.co/docs/datasets/tutorial) | [tutorial_grpo.py#L54-L60](scripts/rl/tutorial_grpo.py) - loading parquet |
| `dataset.map()` | HF Datasets processing | [chat_dataset.py#L98](phitrain/datasets/rl/chat/chat_dataset.py) - `_tokenize_dataset()` |

**Key concepts to understand**:
- Difference between `tokenize=True` vs `tokenize=False` in `apply_chat_template()`
- What `add_generation_prompt=True` does (adds the assistant turn start)
- How chat messages format: `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`
- Why `add_special_tokens=False` when tokenizing already-templated text
- Lazy vs eager loading in HF datasets

**Practice**:
```python
from transformers import AutoTokenizer
from datasets import load_dataset

# 1. Load tokenizer and explore chat template
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

messages = [
    {"role": "user", "content": "What is 2+2?"},
]

# 2. Compare tokenize=True vs tokenize=False
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("Raw text:\n", text)

token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
print("Token count:", len(token_ids))
print("Decoded:", tokenizer.decode(token_ids))

# 3. Load and process a dataset
dataset = load_dataset("gsm8k", "main", split="train[:100]")
print("Columns:", dataset.column_names)

def add_prompt(row):
    row["messages"] = [{"role": "user", "content": row["question"]}]
    return row

dataset = dataset.map(add_prompt)
print("First example messages:", dataset[0]["messages"])
```

**Key files to study**:
- [chat_dataset.py](phitrain/datasets/rl/chat/chat_dataset.py) - Full dataset pipeline
- [rl_data_collator.py](phitrain/datasets/rl/rl_data_collator.py) - Batching logic

---

## **Day 2: Ray + vLLM (The Distributed Stack)**

### Morning: **Ray Core**

| Topic | Resource | Used In Library |
|-------|----------|-----------------|
| `ray.init()` | [Ray Core Quickstart](https://docs.ray.io/en/latest/ray-core/walkthrough.html) | [ray_tuner.py#L47](phitrain/rl/tuners/ray_tuner.py) - `if not ray.is_initialized(): ray.init()` |
| `@ray.remote` decorators | Ray Actors Guide | [ray_worker.py#L80](phitrain/rl/tuners/ray_worker.py) - `ray.remote(num_gpus=1)(cls)` |
| `ray.get()` for results | Ray Actors Guide | [grpo_tuner.py#L84-L88](phitrain/rl/tuners/grpo/grpo_tuner.py) - gathering worker results |
| `.remote()` calls | Ray Actors Guide | [grpo_tuner.py#L82-L84](phitrain/rl/tuners/grpo/grpo_tuner.py) - `worker.update_actor_policy.remote()` |
| Actor spawning across GPUs | Ray Cluster docs | [ray_worker.py#L64-L90](phitrain/rl/tuners/ray_worker.py) - `spawn_all()` |
| Actor naming | Ray docs | [distributed_layout.py#L60-L70](phitrain/rl/distributed_layout.py) - `worker_names` property |

**Key concepts to understand**:
- Ray Actor = a stateful class running on a remote worker (like `RayGRPOWorker`)
- `.remote()` returns a future (ObjectRef), not the actual result
- `ray.get()` blocks until the future resolves
- `num_gpus=1` reserves a GPU for that actor
- Actors can call methods on each other via their handles (`ActorHandle`)
- Ray namespace isolates actors across different runs

**Practice**:
```python
import ray

ray.init()

# 1. Create a simple actor (like a mini RayWorker)
@ray.remote(num_gpus=0)  # use num_gpus=1 if you have GPUs
class ModelWorker:
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.step = 0
    
    def train_step(self, batch_data: list) -> dict:
        self.step += 1
        # Simulate training
        loss = sum(batch_data) / len(batch_data)
        return {"worker_id": self.worker_id, "step": self.step, "loss": loss}
    
    def get_state(self) -> dict:
        return {"worker_id": self.worker_id, "step": self.step}

# 2. Spawn multiple workers (like spawn_all in ray_worker.py)
workers = [ModelWorker.remote(i) for i in range(4)]

# 3. Call methods in parallel (like grpo_tuner.py does)
batch = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
futures = [w.train_step.remote(batch[i]) for i, w in enumerate(workers)]

# 4. Gather results (this is what ray.get does)
results = ray.get(futures)
print("Results from all workers:", results)

# 5. Aggregate metrics (like in grpo_tuner._update_actor_policy)
avg_loss = sum(r["loss"] for r in results) / len(results)
print(f"Average loss across workers: {avg_loss}")

ray.shutdown()
```

**Key files to study**:
- [ray_worker.py](phitrain/rl/tuners/ray_worker.py) - Base worker class with `spawn_all()`
- [grpo_worker.py](phitrain/rl/tuners/grpo/grpo_worker.py) - GRPO-specific worker logic
- [ray_utils.py](phitrain/rl/ray_utils.py) - Helper utilities for Ray

---

### Afternoon: **vLLM**

| Topic | Resource | Used In Library |
|-------|----------|-----------------|
| `vllm.LLM` class | [vLLM Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html) | [sync_vllm_worker.py#L16](phitrain/rl/rollout/sync_vllm_worker.py) - `from vllm import LLM` |
| `SamplingParams` | Sampling docs | [vllm_worker_config.py#L90-L95](phitrain/rl/rollout/vllm_worker_config.py) - `sampling_params` config |
| `RequestOutput` handling | vLLM outputs | [rollout_client.py#L16](phitrain/rl/rollout/rollout_client.py) - `from vllm import RequestOutput` |
| GPU memory management | vLLM config | [vllm_worker_config.py#L57-L62](phitrain/rl/rollout/vllm_worker_config.py) - `gpu_memory_utilization` |
| Prefix caching | vLLM optimization | [tutorial_grpo.py#L35](scripts/rl/tutorial_grpo.py) - `enable_prefix_caching=True` |
| Log probabilities | vLLM output | [base.py#L20-L28](phitrain/rl/agents/base.py) - `_extract_chosen_logprobs()` |

**Key concepts to understand**:
- vLLM uses PagedAttention for efficient KV cache management
- `gpu_memory_utilization=0.5` means reserve 50% of GPU for KV cache
- `temperature=1.0` for training (exploration), `temperature=0` for greedy eval
- `logprobs=True` returns log probabilities needed for policy gradient
- Prefix caching reuses KV cache for shared prompt prefixes
- `enforce_eager=True` disables CUDA graphs (slower but more flexible)

**Practice**:
```python
from vllm import LLM, SamplingParams

# 1. Initialize vLLM (use a small model for testing)
llm = LLM(
    model="microsoft/Phi-3-mini-128k-instruct",
    gpu_memory_utilization=0.5,
    dtype="bfloat16",
    enforce_eager=True,  # For debugging
)

# 2. Configure sampling (like VLLMWorkerConfig.sampling_params)
sampling_params = SamplingParams(
    temperature=1.0,
    max_tokens=256,
    logprobs=1,  # Return log probs for RL training
)

# 3. Generate completions
prompts = [
    "What is 2 + 2? Answer:",
    "What is 3 * 4? Answer:",
]
outputs = llm.generate(prompts, sampling_params)

# 4. Extract results (like rollout_client does)
for output in outputs:
    prompt = output.prompt
    for completion in output.outputs:
        print(f"Prompt: {prompt}")
        print(f"Generated: {completion.text}")
        print(f"Token IDs: {completion.token_ids[:10]}...")
        print(f"Log probs: {[lp[completion.token_ids[i]].logprob for i, lp in enumerate(completion.logprobs[:5])]}")
        print()

# 5. Generate multiple completions per prompt (like group_size in GRPO)
sampling_params_grouped = SamplingParams(
    temperature=1.0,
    max_tokens=64,
    n=4,  # Generate 4 completions per prompt
)
outputs = llm.generate(["Solve: 5 + 7 = "], sampling_params_grouped)
for i, completion in enumerate(outputs[0].outputs):
    print(f"Completion {i}: {completion.text}")
```

**Key files to study**:
- [vllm_worker_config.py](phitrain/rl/rollout/vllm_worker_config.py) - All vLLM configuration options
- [sync_vllm_worker.py](phitrain/rl/rollout/sync_vllm_worker.py) - Synchronous generation
- [rollout_client.py](phitrain/rl/rollout/rollout_client.py) - Client interface for rollouts

---

## **Day 3: RL Concepts + Integration**

### Morning: **RLHF & GRPO Theory**

| Topic | Resource | Used In Library |
|-------|----------|-----------------|
| Policy model (Actor) | RLHF overview | [actor.py](phitrain/rl/models/actor.py) - `class Actor` |
| Reference model | RLHF overview | [reference.py](phitrain/rl/models/reference.py) - `class Reference` |
| KL divergence penalty | PPO/GRPO papers | [grpo_worker.py#L84-L90](phitrain/rl/tuners/grpo/grpo_worker.py) - KL computation in `compute_loss()` |
| Advantage calculation | GRPO paper | [algo_utils.py#L14-L46](phitrain/rl/tuners/algo_utils.py) - `calculate_advantages()` |
| Clipped objectives (ε) | PPO paper | [grpo_worker.py#L70-L75](phitrain/rl/tuners/grpo/grpo_worker.py) - `epsilon_low`, `epsilon_high` |
| Entropy regularization | RL theory | [grpo_worker.py#L96-L97](phitrain/rl/tuners/grpo/grpo_worker.py) - `entropy_coeff` |
| Group-based rewards | GRPO paper | [grpo_config.py#L5](phitrain/rl/tuners/grpo/grpo_config.py) - `group_size` |

**Key concepts to understand**:
- **Policy model (π)**: The model being trained to generate good responses
- **Reference model (π_ref)**: Frozen copy of initial policy to prevent drift
- **Advantage**: `A = reward - mean_reward_per_prompt` (how much better than average)
- **GRPO vs PPO**: GRPO uses group-relative advantages, no critic/value network
- **KL penalty**: `β * KL(π || π_ref)` keeps policy close to reference
- **Clipping**: `clip(ratio, 1-ε, 1+ε)` prevents too-large policy updates
- **Why group_size matters**: More samples per prompt → better advantage estimates

**Practice**:
```python
import torch

# 1. Simulate reward calculation for a batch
# Batch: 4 prompts, 8 completions each (group_size=8)
rewards = torch.tensor([
    [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # Prompt 1: 2 correct
    [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # Prompt 2: 4 correct
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Prompt 3: 0 correct
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Prompt 4: 8 correct
])

# 2. Calculate advantages (like algo_utils.calculate_advantages)
mean_per_prompt = rewards.mean(dim=1, keepdim=True)
advantages = rewards - mean_per_prompt
print("Mean reward per prompt:", mean_per_prompt.squeeze())
print("Advantages:\n", advantages)

# 3. Normalize advantages per prompt (GRPO style)
std_per_prompt = rewards.std(dim=1, keepdim=True) + 1e-4
normalized_advantages = advantages / std_per_prompt
print("Normalized advantages:\n", normalized_advantages)

# 4. Simulate clipped policy loss
old_logprobs = torch.randn(4, 8) * 0.1 - 2.0  # log probs from rollout
new_logprobs = old_logprobs + torch.randn(4, 8) * 0.05  # slightly updated policy

ratio = torch.exp(new_logprobs - old_logprobs)
epsilon = 0.2
clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

loss1 = -ratio * normalized_advantages
loss2 = -clipped_ratio * normalized_advantages
loss = torch.max(loss1, loss2).mean()

print(f"Policy loss: {loss.item():.4f}")
print(f"Clip fraction: {(ratio != clipped_ratio).float().mean().item():.2%}")

# 5. Add KL penalty (like grpo_worker.py)
ref_logprobs = old_logprobs.detach()  # Reference model logprobs
kl = ref_logprobs - new_logprobs  # Simplified KL
kl_penalty = torch.clamp(torch.exp(kl) - kl - 1, min=0).mean()
beta = 0.001
total_loss = loss + beta * kl_penalty
print(f"KL penalty: {kl_penalty.item():.4f}")
print(f"Total loss: {total_loss.item():.4f}")
```

**Key config file**: [grpo_config.py](phitrain/rl/tuners/grpo/grpo_config.py) - All GRPO hyperparameters explained in docstrings

---

### Afternoon: **Code Walkthrough**

| Component | Files to Trace |
|-----------|---------------|
| **Entry point** | [tutorial_grpo.py](scripts/rl/tutorial_grpo.py) |
| **Tuner initialization** | [grpo_tuner.py](phitrain/rl/tuners/grpo/grpo_tuner.py) → [ray_tuner.py](phitrain/rl/tuners/ray_tuner.py) |
| **Worker spawning** | [ray_worker.py#L64-L90](phitrain/rl/tuners/ray_worker.py) `spawn_all()` |
| **Rollout generation** | [rollout_client.py](phitrain/rl/rollout/rollout_client.py) → [sync_vllm_worker.py](phitrain/rl/rollout/sync_vllm_worker.py) |
| **Reward scoring** | [gsm8k.py](phitrain/rl/rewards/gsm8k.py) → [reward_manager.py](phitrain/rl/rewards/reward_manager.py) |
| **Policy update** | [grpo_worker.py#L128-L150](phitrain/rl/tuners/grpo/grpo_worker.py) `update_actor_policy()` |

**Key concepts to understand**:
- The training loop lives in `RayGRPOTuner.train()`, not in individual workers
- Workers are stateful Ray actors holding model shards
- Rollouts happen via vLLM (separate from training model)
- Weights are synced from Actor → vLLM after each update
- `PackedBatch` efficiently packs variable-length sequences

**Practice**:
```python
# Trace through the tutorial step by step with print statements

# 1. Add debug prints to understand the flow
# In tutorial_grpo.py, before tuner.train():
print(f"Tuner config: {tuning_args.to_dict()}")
print(f"Train dataset size: {len(train_dataset)}")
print(f"Eval dataset size: {len(eval_dataset)}")

# 2. Examine a single training sample
sample = train_dataset[0]
print("Sample keys:", sample.keys())
print("Prompt tokens:", len(sample["prompt_token_ids"]))
print("Ground truth:", sample["ground_truth"])

# 3. Trace the reward function
from phitrain.rl.rewards.gsm8k import GSM8kReward
reward = GSM8kReward(format_score=0.0, correct_score=1.0)

test_solution = "Let me solve this step by step.\n2 + 2 = 4\n#### 4"
test_ground_truth = "4"
result = reward.score(test_solution, test_ground_truth)
print(f"Reward score: {result.score}, Correct: {result.correct}")

# 4. Understand the distributed layout
from phitrain.rl.distributed_layout import DistributedLayout
layout = DistributedLayout(n_nodes=1, n_gpus_per_node=4, actor_tp_size=1, rollout_tp_size=1)
print(f"World size: {layout.world_size}")
print(f"Actor DP size: {layout.actor_dp_size}")
print(f"Worker names: {layout.worker_names}")
```

---

## **File Dependency Graph**

```
tutorial_grpo.py
├── RayGRPOConfig (grpo_config.py)
│   ├── ActorConfig (actor_config.py)
│   └── VLLMWorkerConfig (vllm_worker_config.py)
├── RayGRPOTuner (grpo_tuner.py)
│   ├── RayTuner (ray_tuner.py)
│   │   ├── DistributedLayout (distributed_layout.py)
│   │   ├── RewardManager (reward_manager.py)
│   │   └── TunerSchedule (tuner_schedule.py)
│   ├── RayGRPOWorker (grpo_worker.py)
│   │   ├── RayWorker (ray_worker.py)
│   │   ├── Actor (actor.py) ──► FSDP (data_parallel.py)
│   │   ├── Reference (reference.py)
│   │   └── SyncVLLMWorker (sync_vllm_worker.py)
│   └── RolloutClient (rollout_client.py)
├── ChatDataset (chat_dataset.py)
├── ChatDataCollator (rl_data_collator.py)
└── GSM8kReward (gsm8k.py)
    └── Reward (reward.py)
```

---

## **Quick Lookup: Where Each Concept Lives**

| Concept | Primary File(s) |
|---------|-----------------|
| Ray actor spawning | [ray_worker.py#L64-L90](phitrain/rl/tuners/ray_worker.py) |
| FSDP sharding | [data_parallel.py](phitrain/models/parallellisms/data_parallel.py) |
| Device mesh creation | [distributed_layout.py#L150-L200](phitrain/rl/distributed_layout.py) |
| vLLM generation | [sync_vllm_worker.py](phitrain/rl/rollout/sync_vllm_worker.py) |
| Tokenization | [chat_dataset.py#L84-L100](phitrain/datasets/rl/chat/chat_dataset.py) |
| Reward computation | [reward.py](phitrain/rl/rewards/reward.py), [gsm8k.py](phitrain/rl/rewards/gsm8k.py) |
| Advantage normalization | [algo_utils.py](phitrain/rl/tuners/algo_utils.py) |
| GRPO loss | [grpo_worker.py#L40-L100](phitrain/rl/tuners/grpo/grpo_worker.py) |
| Sequence packing | [packing.py](phitrain/datasets/rl/packing.py) |
| Log probabilities | [logits_utils.py](phitrain/rl/models/logits_utils.py) |
| Optimizer setup | [actor.py#L125-L150](phitrain/rl/models/actor.py), [registry.py](phitrain/optimizers/registry.py) |
| Weights sync (actor→vLLM) | [sync_vllm_worker.py#L36-L80](phitrain/rl/rollout/sync_vllm_worker.py) |
| wandb logging | [ray_utils.py#L48-L62](phitrain/rl/ray_utils.py) |
| Checkpointing | [ray_tuner.py#L130-L150](phitrain/rl/tuners/ray_tuner.py) |

---

## **Priority Matrix (with Files)**

| Must Know | File Reference |
|-----------|----------------|
| Ray actors/remote | [ray_worker.py](phitrain/rl/tuners/ray_worker.py), [grpo_tuner.py](phitrain/rl/tuners/grpo/grpo_tuner.py) |
| vLLM generation | [sync_vllm_worker.py](phitrain/rl/rollout/sync_vllm_worker.py), [rollout_client.py](phitrain/rl/rollout/rollout_client.py) |
| Tokenization | [chat_dataset.py](phitrain/datasets/rl/chat/chat_dataset.py) |
| GRPO loss math | [grpo_worker.py#L40-L100](phitrain/rl/tuners/grpo/grpo_worker.py) |
| Reward concept | [reward.py](phitrain/rl/rewards/reward.py), [gsm8k.py](phitrain/rl/rewards/gsm8k.py) |

| Should Know | File Reference |
|-------------|----------------|
| FSDP details | [data_parallel.py](phitrain/models/parallellisms/data_parallel.py), [actor.py](phitrain/rl/models/actor.py) |
| Device Mesh | [distributed_layout.py](phitrain/rl/distributed_layout.py) |
| Gradient clipping | [grpo_worker.py](phitrain/rl/tuners/grpo/grpo_worker.py), [parallel_utils.py](phitrain/models/parallellisms/parallel_utils.py) |
| KL penalty | [grpo_worker.py#L84-L90](phitrain/rl/tuners/grpo/grpo_worker.py) |
