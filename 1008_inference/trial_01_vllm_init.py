from vllm import LLM, SamplingParams

def hello_world_vllm():
    """
    Simple vLLM hello world example.
    This demonstrates basic text generation using vLLM.
    """
    
    # Initialize the LLM
    # Using a small model for quick testing - you can change this to other models
    # Popular options: "microsoft/DialoGPT-medium", "gpt2", "facebook/opt-125m"
    model_name = "gpt2"
    
    print(f"Loading model: {model_name}")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,  # Number of GPUs to use
        max_model_len=512,       # Maximum sequence length
    )
    
    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,    # Controls randomness (0.0 = deterministic, 1.0 = very random)
        top_p=0.95,        # Nucleus sampling parameter
        max_tokens=50,     # Maximum number of tokens to generate
        repetition_penalty=1.1  # Penalty for repeating tokens
    )
    
    # Input prompts
    prompts = [
        "Hello, world! This is a simple example of",
        "The future of artificial intelligence is",
        "In a world where technology advances rapidly,",
        "Once upon a time, in a small village,"
    ]
    
    print("\n" + "="*60)
    print("VLLM HELLO WORLD EXAMPLE")
    print("="*60)
    
    # Generate text
    print(f"\nGenerating text with {model_name}...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Print results
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        
        print(f"\n--- Example {i+1} ---")
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        print(f"Full text: {prompt}{generated_text}")
    
    print("\n" + "="*60)
    print("Generation complete!")

def simple_chat_example():
    """
    A simple chat-like example with vLLM.
    """
    model_name = "gpt2"
    
    llm = LLM(model=model_name, max_model_len=256)
    
    # More focused sampling for chat
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=30,
        repetition_penalty=1.2
    )
    
    print("\n" + "="*40)
    print("SIMPLE CHAT EXAMPLE")
    print("="*40)
    
    chat_prompts = [
        "Human: Hello! How are you?\nAssistant:",
        "Human: What's the weather like?\nAssistant:",
        "Human: Tell me a joke.\nAssistant:",
    ]
    
    outputs = llm.generate(chat_prompts, sampling_params)
    
    for output in outputs:
        print(f"\n{output.prompt}{output.outputs[0].text}")

def batch_generation_example():
    """
    Example showing batch generation capabilities of vLLM.
    """
    model_name = "gpt2"
    
    llm = LLM(model=model_name, max_model_len=128)
    
    # Different sampling parameters for variety
    sampling_params = SamplingParams(
        temperature=0.9,
        max_tokens=25,
        n=2,  # Generate 2 outputs per prompt
    )
    
    print("\n" + "="*50)
    print("BATCH GENERATION EXAMPLE (Multiple outputs)")
    print("="*50)
    
    prompts = [
        "The quick brown fox",
        "Machine learning is",
        "Python programming"
    ]
    
    outputs = llm.generate(prompts, sampling_params)
    
    for i, output in enumerate(outputs):
        print(f"\n--- Prompt {i+1}: {output.prompt} ---")
        for j, generated_output in enumerate(output.outputs):
            print(f"  Output {j+1}: {generated_output.text}")

if __name__ == "__main__":
    try:
        # Run the hello world example
        hello_world_vllm()
        
        # Run additional examples
        simple_chat_example()
        batch_generation_example()
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have vLLM installed:")
        print("pip install vllm")
        print("\nOr if you're using CUDA:")
        print("pip install vllm[cuda]")

        