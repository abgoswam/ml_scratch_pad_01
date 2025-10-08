from vllm import LLM, SamplingParams

# Absolutely minimal vLLM example
print("Loading model...")
llm = LLM(model="gpt2")

print("Generating...")
outputs = llm.generate(["Hello world"], SamplingParams(max_tokens=20))

print("Result:")
print(outputs[0].outputs[0].text)