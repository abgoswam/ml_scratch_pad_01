import torch
import time
import numpy as np
from tqdm import tqdm

# Use theoretical peak of ~3.0 TFLOP/s
gtx1650_tflop_per_sec = 3.0

# Matrix sizes to test
sizes = [512, 1024, 2048, 4096,
         8192, 16_384, 32_768]

for N in tqdm(sizes, desc="Matrix sizes"):
    # Random NxN matrix on GPU in FP32 (better for GTX 1650)
    x = torch.randn(N, N, device="cuda", dtype=torch.float32)

    # FLOPs for matrix multiply: 2 * N^3
    flops = 2 * N * N * N

    # Warm up GPU
    for _ in range(10):
        _ = x @ x
    
    # Synchronize GPU before timing
    torch.cuda.synchronize()
    
    times = []
    for i in tqdm(range(100), desc=f"N={N}", leave=False):
        torch.cuda.synchronize()  # Ensure GPU is ready
        t0 = time.perf_counter()  # More precise timer
        y = x @ x
        torch.cuda.synchronize()  # Wait for GPU to finish
        times.append(time.perf_counter() - t0)

    sec = np.mean(times)
    tflops_per_sec = flops / sec / 1e12
    hfu = 100 * tflops_per_sec / gtx1650_tflop_per_sec

    print(f"\nN: {N}, "
          f"FLOP: {flops:.2e}, "
          f"Time: {sec:.6f}s, "
          f"TFLOP/sec: {tflops_per_sec:.2f}, "
          f"HFU: {hfu:.2f}%")
