What is "sharding" and why FSDP shards model parameters across GPUs
Difference between Data Parallel (DP), Tensor Parallel (TP), and Pipeline Parallel
How DeviceMesh organizes GPUs into a logical grid (e.g., 2 nodes × 4 GPUs)
Why reshard_after_forward=True saves memory but costs communication
Mixed precision: why train in fp32 but compute in bf16