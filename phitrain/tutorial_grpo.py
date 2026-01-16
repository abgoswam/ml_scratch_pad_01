import os

import datasets
from transformers import AutoTokenizer

from phitrain.datasets.rl.chat.chat_dataset import ChatDataset
from phitrain.datasets.rl.rl_data_collator import ChatDataCollator
from phitrain.rl.tuners.grpo.grpo_config import RayGRPOConfig
from phitrain.rl.tuners.grpo.grpo_tuner import RayGRPOTuner
from phitrain.rl.models.actor_config import ActorConfig
from phitrain.rl.rollout.vllm_worker_config import VLLMWorkerConfig
from phitrain.rl.rewards.gsm8k import GSM8kReward

os.environ["WANDB_MODE"] = "disabled"

actor_config = ActorConfig(
    model={"pretrained_model_name_or_path": "/home/agoswami/_hackerreborn/aifsdk/_ckpt/mixformer/"},
    optimizer={
        "betas": [0.9, 0.999],
        "weight_decay": 0.01,
    },
    scheduler={
        "type": "warmup",
        "warmup_num_steps": 1,
        "warmup_max_lr": 5.0e-6,
    }
)

rollout_config = VLLMWorkerConfig(
    prompt_length=256,
    response_length=512,
    dtype="bfloat16",
    gpu_memory_utilization=0.5,
    enforce_eager=False,
    enable_prefix_caching=True,
    sampling_params={"temperature": 1.0},
)

tuning_args = RayGRPOConfig(
    output_dir="/tmp/grpo_gsm8k",
    n_nodes=1,
    n_gpus_per_node=4,
    max_steps=1,
    train_batch_size=16,
    group_size=8,
    train_max_micro_batch_size_per_gpu=1,
    actor=actor_config,
    rollout=rollout_config,
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

def _extract_answer(row):
    row["answer"] = row["reward_model"]["ground_truth"]
    return row

dataset = datasets.load_dataset(
    "parquet",
    data_files={
        "train": "/home/gderosa/datasets/gsm8k/train.parquet",
        "test": "/home/gderosa/datasets/gsm8k/test.parquet"
    }
)
dataset = dataset.map(_extract_answer)

reward = GSM8kReward(format_score=0.0, correct_score=1.0)

train_dataset = ChatDataset(
    dataset["train"],
    tokenizer=tokenizer,
    messages_column_name="prompt",
    ground_truth_column_name="answer",
    max_length=tuning_args.rollout.prompt_length,
    filter_max_length=True,
)
eval_dataset = ChatDataset(
    dataset["test"],
    tokenizer=tokenizer,
    messages_column_name="prompt",
    ground_truth_column_name="answer",
    max_length=tuning_args.rollout.prompt_length,
)
data_collator = ChatDataCollator()

tuner = RayGRPOTuner(
    args=tuning_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    reward=reward,
)
tuner.train()