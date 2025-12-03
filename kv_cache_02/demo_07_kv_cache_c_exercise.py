#!/usr/bin/env python3
import math, time, torch
import torch.nn.functional as F

torch.manual_seed(42)

class TinySelfAttention(torch.nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.D = D; 
        self.H = H; 
        self.dh = D // H
        
        self.Wq = torch.nn.Linear(D, D)
        self.Wk = torch.nn.Linear(D, D)
        self.Wv = torch.nn.Linear(D, D)
        self.Wo = torch.nn.Linear(D, D)

    def forward(self, x, cache=None):
        T, D = x.shape
        # TODO: Compute Q matrix from input x
        # Hint: Apply Wq transformation and reshape for multi-head attention
        Q = # TODO
        
        if cache is None:
            # Prefill: need causal mask
            # TODO: Compute K and V matrices from input x
            # Hint: Apply Wk and Wv transformations and reshape for multi-head attention
            K = # TODO
            V = # TODO
            
            # TODO: Compute attention scores
            # Hint: Matrix multiply Q and K, don't forget scaling factor
            scores = # TODO
            
            # Apply causal mask for prefill
            if T > 1:
                causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
                scores.masked_fill_(causal_mask, float('-inf'))
        else:
            # Generation: T=1, naturally causal via cache structure
            assert T==1
            # TODO: Compute new k and v for the current token
            # Hint: Only process the last token (x[-1:])
            k_new = # TODO
            v_new = # TODO
            
            if cache["k"] is None:
                K, V = k_new, v_new
            else:
                # TODO: Concatenate cached K,V with new k,v
                # Hint: Use torch.cat along the sequence dimension
                K = # TODO
                V = # TODO
            
            # TODO: Update cache with new K,V
            cache["k"], cache["v"] = # TODO

            # TODO: Compute attention scores for generation
            scores = # TODO
            # No mask needed here!

        # TODO: Apply softmax to get attention probabilities
        P = # TODO
        
        # TODO: Apply attention to values
        Y = # TODO
        
        # TODO: Reshape Y back to original format [T, D]
        Y = # TODO
        
        return self.Wo(Y), cache

class TinyLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size, D, H):
        super().__init__()
        self.vocab_size = vocab_size
        self.D = D
        
        # Token embeddings
        self.token_embedding = torch.nn.Embedding(vocab_size, D)
        
        # Attention layer
        self.attention = TinySelfAttention(D, H)
        
        # Language model head: convert embeddings back to token logits
        self.lm_head = torch.nn.Linear(D, vocab_size, bias=False)

    def forward(self, token_ids, cache=None):
        # token_ids: [T] - sequence of token indices
        # TODO: Convert token IDs to embeddings: [T] -> [T, D]
        # Hint: Use self.token_embedding, handle both scalar and tensor inputs
        x = # TODO
        
        # Apply attention: [T, D] -> [T, D]
        x, cache = self.attention(x, cache)
        
        # TODO: Convert back to token logits: [T, D] -> [T, vocab_size]
        logits = # TODO
        
        return logits, cache

def sample_next_token(logits, temperature=1.0, top_k=None):
    """Sample next token from logits"""
    # logits: [vocab_size] for the last position
    if temperature == 0:
        return torch.argmax(logits, dim=-1)
    
    # TODO: Apply temperature scaling
    logits = # TODO
    
    if top_k is not None:
        # Keep only top-k tokens
        top_k_logits, top_k_indices = torch.topk(logits, top_k)
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(-1, top_k_indices, top_k_logits)
    
    # TODO: Convert logits to probabilities and sample
    probs = # TODO
    return # TODO

def generate_text(model, initial_tokens, max_new_tokens=10, temperature=1.0):
    """Generate text using KV cache"""
    model.eval()
    
    # Convert to tensor if needed
    if isinstance(initial_tokens, list):
        tokens = torch.tensor(initial_tokens)
    else:
        tokens = initial_tokens.clone()
    
    # TODO: Initialize cache dictionary
    cache = # TODO
    generated_tokens = []
    
    with torch.no_grad():
        # Process initial sequence (prefill)
        if len(tokens) > 1:
            # TODO: Process all tokens except the last one for prefill
            logits, cache = # TODO
        
        # Generate new tokens one by one
        for _ in range(max_new_tokens):
            # TODO: Get logits for the last token using the cache
            logits, cache = # TODO
            
            # Sample next token from the last position: [1, vocab_size] -> [vocab_size]
            next_token = sample_next_token(logits[-1], temperature=temperature)
            
            # TODO: Add new token to the sequence
            tokens = # TODO
            generated_tokens.append(next_token.item())
    
    return tokens, generated_tokens

def main():
    # Model parameters
    vocab_size = 100  # Small vocabulary for demo
    D = 4            # Model dimension
    H = 2            # Number of heads
    
    # Create model
    model = TinyLanguageModel(vocab_size, D, H)
    
    # Initial sequence (e.g., prompt tokens)
    initial_tokens = [1]  # Some token IDs
    
    # Generate text
    print("Generating text with KV cache...")
    full_sequence, new_tokens = generate_text(
        model, 
        initial_tokens, 
        max_new_tokens=5,
        temperature=1.0
    )
    
    print(f"Initial tokens: {initial_tokens}")
    print(f"Generated tokens: {new_tokens}")
    print(f"Full sequence: {full_sequence.tolist()}")

if __name__ == "__main__":
    main()