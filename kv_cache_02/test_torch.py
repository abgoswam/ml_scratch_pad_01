import torch
import sys

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        gpu = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {gpu.name}")
        print(f"  Memory: {gpu.total_memory / 1024**3:.1f} GB")
        print(f"  Compute capability: {gpu.major}.{gpu.minor}")
    
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"Current GPU name: {torch.cuda.get_device_name()}")
    
    # Test GPU tensor operations
    try:
        x = torch.randn(3, 3).cuda()
        y = torch.randn(3, 3).cuda()
        z = torch.matmul(x, y)
        print("✅ GPU tensor operations working")
        print(f"Test tensor device: {z.device}")
    except Exception as e:
        print(f"❌ GPU tensor operations failed: {e}")
        
else:
    print("❌ CUDA not available - using CPU only")
    print("Reasons CUDA might not be available:")
    print("  - No GPU hardware")
    print("  - CUDA drivers not installed")
    print("  - PyTorch installed without CUDA support")
    print("  - Environment/conda issues")

# Check if MPS (Metal Performance Shaders) is available (for Apple Silicon)
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print("✅ MPS (Metal Performance Shaders) available for Apple Silicon")
else:
    print("❌ MPS not available")