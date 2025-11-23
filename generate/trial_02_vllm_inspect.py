from vllm import LLM, SamplingParams
import torch
import os

def hello_world_vllm_conservative():
    """
    Conservative vLLM hello world example that avoids problematic optimizations.
    """
    
    model_name = "gpt2"
    
    print(f"Loading model: {model_name}")
    
    # Use very conservative settings to avoid FlexAttention issues
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        max_model_len=256,           # Smaller context to reduce memory pressure
        enforce_eager=True,          # Disable CUDA graphs
        disable_custom_all_reduce=True,
        gpu_memory_utilization=0.6,  # Conservative memory usage
        enable_chunked_prefill=False,  # Disable chunked prefill
        use_v2_block_manager=False,    # Use legacy block manager
        # Force specific attention backend
        attention_backend="XFORMERS",  # Try XFormers instead of FlexAttention
    )
    
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=30,  # Shorter outputs
        repetition_penalty=1.1
    )
    
    # Simple single prompt first
    prompt = "Hello, world! This is a simple example of"
    
    print("\n" + "="*60)
    print("CONSERVATIVE VLLM HELLO WORLD EXAMPLE")
    print("="*60)
    
    print(f"\nGenerating text with {model_name}...")
    print(f"Prompt: {prompt}")
    
    outputs = llm.generate([prompt], sampling_params)
    
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated: {generated_text}")
        print(f"Full text: {prompt}{generated_text}")
    
    print("\n" + "="*60)
    print("Generation complete!")

def try_different_backends():
    """
    Try different attention backends to find one that works.
    """
    model_name = "gpt2"
    prompt = "Hello, world!"
    
    # Try different attention backends
    backends_to_try = [
        "XFORMERS",
        "FLASH_ATTN", 
        "TORCH_SDPA",
        None,  # Default
    ]
    
    for backend in backends_to_try:
        print(f"\n{'='*50}")
        print(f"TRYING ATTENTION BACKEND: {backend or 'DEFAULT'}")
        print(f"{'='*50}")
        
        try:
            llm_args = {
                "model": model_name,
                "max_model_len": 128,
                "enforce_eager": True,
                "disable_custom_all_reduce": True,
                "gpu_memory_utilization": 0.5,
                "enable_chunked_prefill": False,
                "use_v2_block_manager": False,
            }
            
            if backend:
                llm_args["attention_backend"] = backend
            
            llm = LLM(**llm_args)
            
            sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=20,
            )
            
            outputs = llm.generate([prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text
            
            print(f"✅ SUCCESS with {backend or 'DEFAULT'}")
            print(f"Generated: {generated_text}")
            return llm, backend
            
        except Exception as e:
            print(f"❌ FAILED with {backend or 'DEFAULT'}: {str(e)[:100]}...")
            continue
    
    print("\n❌ All backends failed!")
    return None, None

def fallback_to_legacy_engine():
    """
    Try using vLLM's legacy (v0) engine instead of v1.
    """
    print(f"\n{'='*50}")
    print("TRYING LEGACY ENGINE (V0)")
    print(f"{'='*50}")
    
    # Set environment variable to force legacy engine
    os.environ["VLLM_USE_V1"] = "0"
    
    try:
        llm = LLM(
            model="gpt2",
            max_model_len=256,
            enforce_eager=True,
            gpu_memory_utilization=0.6,
        )
        
        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=30,
        )
        
        prompt = "Hello, world! This is a test of"
        outputs = llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        print(f"✅ SUCCESS with legacy engine!")
        print(f"Generated: {generated_text}")
        return True
        
    except Exception as e:
        print(f"❌ Legacy engine also failed: {str(e)[:100]}...")
        return False
    finally:
        # Reset the environment variable
        if "VLLM_USE_V1" in os.environ:
            del os.environ["VLLM_USE_V1"]

def cpu_only_fallback():
    """
    Last resort: try CPU-only mode.
    """
    print(f"\n{'='*50}")
    print("TRYING CPU-ONLY MODE")
    print(f"{'='*50}")
    
    # Temporarily hide CUDA
    old_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    try:
        llm = LLM(
            model="gpt2",
            max_model_len=128,
            enforce_eager=True,
        )
        
        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=20,
        )
        
        prompt = "Hello, world!"
        outputs = llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        print(f"✅ SUCCESS with CPU-only mode!")
        print(f"Generated: {generated_text}")
        return True
        
    except Exception as e:
        print(f"❌ CPU-only mode failed: {str(e)[:100]}...")
        return False
    finally:
        # Restore CUDA visibility
        os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_visible

if __name__ == "__main__":
    print("Starting vLLM troubleshooting sequence...\n")
    
    # Step 1: Try conservative settings
    try:
        hello_world_vllm_conservative()
        print("✅ Conservative settings worked!")
        exit(0)
    except Exception as e:
        print(f"❌ Conservative settings failed: {str(e)[:100]}...")
    
    # Step 2: Try different attention backends
    try:
        llm, backend = try_different_backends()
        if llm:
            print(f"✅ Found working backend: {backend}")
            exit(0)
    except Exception as e:
        print(f"❌ All attention backends failed: {str(e)[:100]}...")
    
    # Step 3: Try legacy engine
    try:
        if fallback_to_legacy_engine():
            print("✅ Legacy engine worked!")
            exit(0)
    except Exception as e:
        print(f"❌ Legacy engine failed: {str(e)[:100]}...")
    
    # Step 4: Try CPU-only
    try:
        if cpu_only_fallback():
            print("✅ CPU-only mode worked!")
            print("\nNote: You're running on CPU. For GPU inference, you may need:")
            print("1. Different vLLM version")
            print("2. Different PyTorch version") 
            print("3. Different CUDA version")
            exit(0)
    except Exception as e:
        print(f"❌ CPU-only mode failed: {str(e)[:100]}...")
    
    print("\n" + "="*60)
    print("TROUBLESHOOTING COMPLETE - ALL OPTIONS FAILED")
    print("="*60)
    print("\nPossible solutions:")
    print("1. Downgrade vLLM: pip install vllm==0.5.5")
    print("2. Use different model: try 'facebook/opt-125m'")
    print("3. Use transformers instead of vLLM for now")
    print("4. Check vLLM GitHub issues for your specific GPU/CUDA combo")