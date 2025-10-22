import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    // Block and thread indices
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    // Tile size
    const int TILE_SIZE = 32;
    
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Calculate global row and column indices
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        if (row < N && t * TILE_SIZE + tx < N) {
            As[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (col < N && t * TILE_SIZE + ty < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());
    
    const int TILE_SIZE = 32;
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );
    
    return C;
}
"""

matmul_cpp_source = (
    "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for matrix multiplication
matmul = load_inline(
    name="matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a single square matrix multiplication (C = A * B)
    using custom CUDA kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication using custom CUDA kernel.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return self.matmul.matmul_cuda(A, B)

N = 2048 * 2

def get_inputs():
    A = torch.rand(N, N).cuda()
    B = torch.rand(N, N).cuda()
    return [A, B]

# def get_init_inputs():
#     return []  # No special initialization inputs needed

print("done")

# Add actual test execution:
if __name__ == "__main__":
    # Create model
    model = ModelNew()
    
    # Get test inputs
    A, B = get_inputs()
    
    # Run custom kernel
    result_custom = model.forward(A, B)
    
    # Compare with PyTorch's built-in matmul
    result_pytorch = torch.matmul(A, B)
    
    # Check correctness
    print(f"Results match: {torch.allclose(result_custom, result_pytorch, atol=1e-4)}")
    print(f"Max difference: {torch.max(torch.abs(result_custom - result_pytorch))}")

    # benchmark.

    model = ModelNew().cuda()

    # --- setup ---
    torch.manual_seed(0)
    N = 2048  # you can also try 4096 for a heavier test

    A = torch.rand(N, N, device="cuda", dtype=torch.float32)
    B = torch.rand(N, N, device="cuda", dtype=torch.float32)

    # A = torch.rand(N, N, device="cuda", dtype=torch.half)
    # B = torch.rand(N, N, device="cuda", dtype=torch.half)

    # warmup (to stabilize GPU clocks)
    for _ in range(5):
        _ = torch.matmul(A, B)
        _ = model(A, B)

    torch.cuda.synchronize()

    # --- helper for timing ---
    def benchmark(fn, *args, repeat=10):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        total_time = 0.0
        for _ in range(repeat):
            start.record()
            _ = fn(*args)
            end.record()
            torch.cuda.synchronize()
            total_time += start.elapsed_time(end)
        return total_time / repeat  # milliseconds

    # --- benchmark torch.matmul ---
    t_matmul = benchmark(torch.matmul, A, B)
    # --- benchmark custom kernel ---
    t_custom = benchmark(model, A, B)

    # --- correctness check ---
    C_ref = torch.matmul(A, B)
    C_custom = model(A, B)
    max_diff = (C_ref - C_custom).abs().max().item()

    print(f"Matrix size: {N}x{N}")
    print(f"torch.matmul: {t_matmul:.3f} ms")
    print(f"custom kernel: {t_custom:.3f} ms")
    print(f"Speedup: {t_matmul / t_custom:.2f}× (if >1 → faster)")
    print(f"Max difference: {max_diff:.6f}")