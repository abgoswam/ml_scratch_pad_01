from vllm import LLM, SamplingParams

def simple_hello_world():
    """
    Simple vLLM hello world example for v0.5.5 - should work reliably.
    """
    
    print("=" * 60)
    print("vLLM 0.5.5 - Simple Hello World")
    print("=" * 60)
    
    # Initialize the model - v0.5.5 uses simpler, more stable parameters
    model_name = "gpt2"
    
    print(f"Loading model: {model_name}")
    llm = LLM(
        model=model_name,
        max_model_len=512,
        tensor_parallel_size=1,
        trust_remote_code=False,
    )
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=50,
        repetition_penalty=1.1,
    )
    
    # Test prompts
    prompts = [
        "Hello, world! This is a simple example of",
        "The future of artificial intelligence is",
        "Python programming is",
        "Machine learning helps us"
    ]
    
    print(f"\nGenerating text with {model_name}...")
    print("-" * 60)
    
    # Generate responses
    outputs = llm.generate(prompts, sampling_params)
    
    # Display results
    for i, output in enumerate(outputs, 1):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        
        print(f"\nExample {i}:")
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        print(f"Complete: {prompt}{generated_text}")
    
    print("\n" + "=" * 60)
    print("✅ Generation complete!")

def chat_example():
    """
    Simple chat-style example.
    """
    
    print("\n" + "=" * 60)
    print("Chat Example")
    print("=" * 60)
    
    llm = LLM(model="gpt2", max_model_len=256)
    
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=40,
        repetition_penalty=1.2,
    )
    
    chat_prompts = [
        "Human: What is Python?\nAssistant:",
        "Human: How do neural networks work?\nAssistant:",
        "Human: What's the weather like today?\nAssistant:",
    ]
    
    outputs = llm.generate(chat_prompts, sampling_params)
    
    for i, output in enumerate(outputs, 1):
        print(f"\nChat {i}:")
        print(output.prompt + output.outputs[0].text)

def batch_generation_example():
    """
    Demonstrate batch generation with multiple outputs per prompt.
    """
    
    print("\n" + "=" * 60)
    print("Batch Generation - Multiple Outputs")
    print("=" * 60)
    
    llm = LLM(model="gpt2", max_model_len=256)
    
    # Generate multiple outputs per prompt
    sampling_params = SamplingParams(
        temperature=0.9,
        max_tokens=30,
        n=3,  # Generate 3 different outputs per prompt
    )
    
    prompts = [
        "The best thing about programming is",
        "Artificial intelligence will",
    ]
    
    outputs = llm.generate(prompts, sampling_params)
    
    for i, output in enumerate(outputs, 1):
        print(f"\nPrompt {i}: {output.prompt}")
        for j, generated_output in enumerate(output.outputs, 1):
            print(f"  Version {j}: {generated_output.text}")

def different_models_example():
    """
    Try different small models to show flexibility.
    """
    
    print("\n" + "=" * 60)
    print("Different Models Example")
    print("=" * 60)
    
    models_to_try = [
        "gpt2",
        "distilgpt2",  # Smaller, faster version
    ]
    
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=25,
    )
    
    prompt = "The quick brown fox"
    
    for model_name in models_to_try:
        try:
            print(f"\nTrying model: {model_name}")
            llm = LLM(model=model_name, max_model_len=128)
            
            outputs = llm.generate([prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text
            
            print(f"Result: {prompt}{generated_text}")
            
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")

def streaming_like_example():
    """
    Simulate streaming by generating shorter chunks.
    """
    
    print("\n" + "=" * 60)
    print("Progressive Generation (Simulated Streaming)")
    print("=" * 60)
    
    llm = LLM(model="gpt2", max_model_len=256)
    
    prompt = "Once upon a time, in a magical forest"
    current_text = prompt
    
    print(f"Starting prompt: {prompt}")
    print("Progressive generation:")
    print("-" * 40)
    
    # Generate in chunks to simulate streaming
    for i in range(5):
        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=10,  # Short chunks
        )
        
        outputs = llm.generate([current_text], sampling_params)
        new_text = outputs[0].outputs[0].text
        current_text += new_text
        
        print(f"Step {i+1}: {new_text}")
    
    print("-" * 40)
    print(f"Final result: {current_text}")

if __name__ == "__main__":
    try:
        # Run all examples
        simple_hello_world()
        chat_example()
        batch_generation_example()
        different_models_example()
        streaming_like_example()
        
        print("\n🎉 All examples completed successfully!")
        print("vLLM 0.5.5 is working properly!")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("\nIf you're still having issues:")
        print("1. Check that vLLM 0.5.5 installed correctly: pip list | grep vllm")
        print("2. Verify GPU availability: python -c 'import torch; print(torch.cuda.is_available())'")
        print("3. Try CPU-only: CUDA_VISIBLE_DEVICES='' python script.py")