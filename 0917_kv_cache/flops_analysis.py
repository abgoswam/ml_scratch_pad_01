# %% [markdown]
# <table style="width:100%">
# <tr>
# <td style="vertical-align:middle; text-align:left;">
# <font size="2">
# Supplementary code for the <a href="http://mng.bz/orYv">Build a Large Language Model From Scratch</a> book by <a href="https://sebastianraschka.com">Sebastian Raschka</a><br>
# <br>Code repository: <a href="https://github.com/rasbt/LLMs-from-scratch">https://github.com/rasbt/LLMs-from-scratch</a>
# </font>
# </td>
# <td style="vertical-align:middle; text-align:left;">
# <a href="http://mng.bz/orYv"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/cover-small.webp" width="100px"></a>
# </td>
# </tr>
# </table>

# %% [markdown]
# # FLOPS Analysis

# %% [markdown]
# - FLOPs (Floating Point Operations Per Second) measure the computational complexity of neural network models by counting the number of floating-point operations executed
# - High FLOPs indicate more intensive computation and energy consumption

# %%
# pip install -r requirements-extra.txt

# %%
from importlib.metadata import version

pkgs = [
    "thop",
    "torch",
]
for p in pkgs:
    print(f"{p} version: {version(p)}")

# %% [markdown]
# &nbsp;
# # Simple benchmark with fixed batch size

# %% [markdown]
# - forward pass only

# %%
import torch
from thop import profile

# For installation instructions, see:
# https://github.com/rasbt/LLMs-from-scratch/tree/main/pkg
from llms_from_scratch.ch04 import GPTModel


BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 2
input_tensor = torch.randint(0, 50257, (batch_size, 1024)).to(device)

for size in model_configs:
    BASE_CONFIG.update(model_configs[size])

    model = GPTModel(BASE_CONFIG).bfloat16()
    model.to(device)

    # MACS = multiply-accumulate operations
    # MACS are typically counted as two FLOPS (one multiply and one accumulate)
    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    flops = 2*macs
    print(f"{size:18}: {flops:.1e} FLOPS")

    del model
    torch.cuda.empty_cache()

# %% [markdown]
# &nbsp;
# # Simple benchmark with automatic batch size finding

# %% [markdown]
# - forward pass only

# %%
for size in model_configs:
    print(f"\nProcessing {size}")
    config = BASE_CONFIG.copy()
    config.update(model_configs[size])

    min_batch_size = 1
    max_batch_size = None
    max_possible_batch_size = 4096

    while min_batch_size <= max_possible_batch_size:
        batch_size = (min_batch_size + max_possible_batch_size) // 2
        try:
            input_tensor = torch.randint(
                0, config["vocab_size"],
                (batch_size, config["context_length"]),
                device=device
            )

            model = GPTModel(config).bfloat16().to(device)

            # MACS = multiply-accumulate operations
            # MACS are typically counted as two FLOPS (one multiply and one accumulate)
            macs, params = profile(model, inputs=(input_tensor,), verbose=False)
            flops = 2 * macs
            print(f"  Batch size {batch_size}: {flops:.1e} FLOPS")

            # If successful, try a larger batch size
            min_batch_size = batch_size + 1
            max_batch_size = batch_size

            # Clean up
            del model, input_tensor
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                # Try smaller batch size
                max_possible_batch_size = batch_size - 1

                # Clean up
                try:
                    del model, input_tensor
                    torch.cuda.empty_cache()
                except NameError:
                    pass
            else:
                raise e

# %% [markdown]
# &nbsp;
# # Benchmark with automatic batch size finding and Model FLOP Utilization (MFU)

# %% [markdown]
# - Model FLOPs Utilization (MFU) explanation from the [PaLM paper](https://arxiv.org/abs/2204.02311)
# 
# > We propose a new metric for efficiency that is implementation-independent and permits a cleaner comparison of system efficiency, called model FLOPs utilization (MFU). This is the ratio of the observed throughput (tokens-per-second) relative to the theoretical maximum throughput of a system operating at peak FLOPs. Crucially, the “theoretical maximum” throughput only accounts for the required operations to compute the forward+backward passes, and not rematerialization.
# 
# 
# $$\text{MFU} = \frac{\text{Observed Tokens per Second}}{\text{Theoretical Max Tokens per Second}}$$
# 
# where
# 
# $$\text{Theoretical Max Tokens per Second} = \frac{\text{Max FLOPs per Second}}{\text{Total FLOPs per Token}}$$
# 
# and
# 
# $$\text{Tokens per Second} = \frac{\text{Batch Size} \times \text{Sequence Length}}{\text{Total Time}}$$

# %% [markdown]
# - forward and backward pass

# %%
# Theoretical max flops per second provided by the GPU manufacturer

flops_per_second = {
    # https://www.techpowerup.com/gpu-specs/h100-pcie-80-gb.c3899
    "H100": {
        torch.float32: 51.22e12,  # 51.22 TFLOPs for FP32 on NVIDIA H100
        torch.float16: 204.9e12,  # 204.9 TFLOPs for FP16 on NVIDIA H100
        torch.bfloat16: 204.9e12
    },
    # https://www.techpowerup.com/gpu-specs/l4.c4091
    "L4": {
        torch.float32: 30.29e12,  # 30.29 TFLOPs for FP32 on NVIDIA L4
        torch.float16: 30.29e12,  # 30.29 TFLOPs for FP16 on NVIDIA L4
        torch.bfloat16: 30.29e12
    },
    # https://www.techpowerup.com/gpu-specs/tesla-t4.c3316
    "T4": {
        torch.float32: 8.1e12,  # 8.1 TFLOPs for FP32 on NVIDIA T4
        torch.float16: 65.13e12,  # 65.13 TFLOPs for FP16 on NVIDIA T4
        torch.bfloat16: 65.13e12
    },
    # https://www.techpowerup.com/gpu-specs/a10g.c3798
    "A10G": {
        torch.float32: 31.52e12,  # 31.52 TFLOPs for FP32 on NVIDIA A10G
        torch.float16: 31.52e12,  # 31.52 TFLOPs for FP16 on NVIDIA A10G
        torch.bfloat16: 31.52e12
    },
    # https://www.techpowerup.com/gpu-specs/a100-pcie-40-gb.c3623
    "A100": {
        torch.float32: 19.49e12,  # 19.49 TFLOPs for FP32 on NVIDIA A100
        torch.float16: 77.97e12,  # 77.97 TFLOPs for FP16 on NVIDIA A100
        torch.bfloat16: 77.97e12
    },
    # https://www.techpowerup.com/gpu-specs/geforce-rtx-3080.c3621
    "RTX_3080": {
        torch.float32: 29.77e12,  # 29.77 TFLOPs for FP32 on NVIDIA RTX 3080
        torch.float16: 29.77e12,  # 29.77 TFLOPs for FP16 on NVIDIA RTX 3080
        torch.bfloat16: 29.77e12
    },
    # https://www.techpowerup.com/gpu-specs/geforce-rtx-3090.c3622
    "RTX_3090": {
        torch.float32: 35.58e12,  # 35.58 TFLOPs for FP32 on NVIDIA RTX 3090
        torch.float16: 35.58e12,  # 35.58 TFLOPs for FP16 on NVIDIA RTX 3090
        torch.bfloat16: 35.58e12
    }
}


# %%
import time

def get_gpu_model(flops_per_second_dict):
    device_name = torch.cuda.get_device_name(0)
    for model in flops_per_second_dict.keys():
        if model in device_name:
            return model
    return "Unknown"  # Default if no matching model is found


gpu_model = get_gpu_model(flops_per_second)
print("GPU Model:", gpu_model)

if gpu_model != "Unknown":

    for size in model_configs:
        print(f"\nProcessing {size}")
        config = BASE_CONFIG.copy()
        config.update(model_configs[size])

        min_batch_size = 1
        max_batch_size = None
        max_possible_batch_size = 4096

        while min_batch_size <= max_possible_batch_size:
            batch_size = (min_batch_size + max_possible_batch_size) // 2
            try:
                input_tensor = torch.randint(
                    0, config["vocab_size"],
                    (batch_size, config["context_length"]),
                    device=device
                )

                model = GPTModel(config).bfloat16().to(device)
                model.train()

                # Start timing
                torch.cuda.synchronize()
                start_time = time.time()

                # Forward & backward pass
                output = model(input_tensor)
                loss = output.sum()  # Compute a dummy loss
                loss.backward()

                # End timing
                torch.cuda.synchronize()
                end_time = time.time()

                total_time_seconds = end_time - start_time

                # Calculate FLOPs for forward pass
                macs, params = profile(model, inputs=(input_tensor,), verbose=False)
                flops_forward = 2 * macs  # Assuming one MAC equals two FLOPs

                # Estimate FLOPs for backward pass (typically 2x forward FLOPs)
                flops_backward = 2 * flops_forward

                # Total FLOPs for forward + backward passes
                total_flops = flops_forward + flops_backward  # Or total_flops = flops_forward * 3

                data_type = next(model.parameters()).dtype
                max_flops_per_second = flops_per_second[gpu_model].get(data_type, 0)

                # Compute tokens per second
                tokens_processed = batch_size * config["context_length"]
                tokens_per_second = tokens_processed / total_time_seconds

                # Compute FLOPs per token
                flops_per_token = total_flops / tokens_processed

                # Compute theoretical max tokens per second
                if flops_per_token > 0:
                    theoretical_max_tokens_per_second = max_flops_per_second / flops_per_token
                else:
                    theoretical_max_tokens_per_second = 0  # Avoid division by zero

                # Compute MFU
                if theoretical_max_tokens_per_second > 0:
                    mfu = tokens_per_second / theoretical_max_tokens_per_second
                else:
                    mfu = 0  # Avoid division by zero

                print(f"  Batch size {batch_size}: Tokens/sec: {tokens_per_second:.2f}, MFU: {mfu:.4f}")

                # If successful, try a larger batch size
                min_batch_size = batch_size + 1
                max_batch_size = batch_size

                # Clean up
                del model, input_tensor, output, loss
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Try smaller batch size
                    max_possible_batch_size = batch_size - 1

                    # Clean up
                    try:
                        del model, input_tensor
                        torch.cuda.empty_cache()
                    except NameError:
                        pass
                else:
                    raise e

else:
    print("Unknown GPU model. Please update the flops_per_second dictionary with your GPU information.")

# %% [markdown]
# - a value of 1.0 is best (equal to 100%)
# - Note that the batch sizes are smaller than previously because we also carry out the backward pass here, which is more memory-intensive


