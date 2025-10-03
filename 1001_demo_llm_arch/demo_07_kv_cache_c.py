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
        Q = self.Wq(x).view(T, self.H, self.dh).transpose(0,1)

        if cache is None:
            # Prefill: need causal mask
            K = self.Wk(x).view(T, self.H, self.dh).transpose(0,1)
            V = self.Wv(x).view(T, self.H, self.dh).transpose(0,1)
            
            scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.dh)
            
            # Apply causal mask for prefill
            if T > 1:
                causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
                scores.masked_fill_(causal_mask, float('-inf'))
        else:
            # Generation: T=1, naturally causal via cache structure
            assert T==1
            k_new = self.Wk(x[-1:]).view(1, self.H, self.dh).transpose(0,1)
            v_new = self.Wv(x[-1:]).view(1, self.H, self.dh).transpose(0,1)
            if cache["k"] is None:
                K, V = k_new, v_new
            else:
                K = torch.cat([cache["k"], k_new], dim=1)
                V = torch.cat([cache["v"], v_new], dim=1)
            cache["k"], cache["v"] = K, V

            scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.dh)
            # No mask needed here!

        P = torch.softmax(scores, dim=-1)
        Y = torch.matmul(P, V)
        Y = Y.transpose(0,1).contiguous().view(T, D)
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
        # Convert token IDs to embeddings: [T] -> [T, D]
        x = self.token_embedding(token_ids.unsqueeze(0) if token_ids.dim() == 0 else token_ids)
        
        # Apply attention: [T, D] -> [T, D]
        x, cache = self.attention(x, cache)
        
        # Convert back to token logits: [T, D] -> [T, vocab_size]
        logits = self.lm_head(x)
        
        return logits, cache

def sample_next_token(logits, temperature=1.0, top_k=None):
    """Sample next token from logits"""
    # logits: [vocab_size] for the last position
    if temperature == 0:
        return torch.argmax(logits, dim=-1)
    
    logits = logits / temperature
    
    if top_k is not None:
        # Keep only top-k tokens
        top_k_logits, top_k_indices = torch.topk(logits, top_k)
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(-1, top_k_indices, top_k_logits)
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).squeeze()

def generate_text(model, initial_tokens, max_new_tokens=10, temperature=1.0):
    """Generate text using KV cache"""
    model.eval()
    
    # Convert to tensor if needed
    if isinstance(initial_tokens, list):
        tokens = torch.tensor(initial_tokens)
    else:
        tokens = initial_tokens.clone()
    
    cache = {"k": None, "v": None}
    generated_tokens = []
    
    with torch.no_grad():
        # Process initial sequence (prefill)
        if len(tokens) > 1:
            logits, cache = model(tokens[:-1], cache=cache)
        
        # Generate new tokens one by one
        for _ in range(max_new_tokens):
            # Get logits for the last token
            logits, cache = model(tokens[-1:], cache=cache)
            
            # Sample next token from the last position: [1, vocab_size] -> [vocab_size]
            next_token = sample_next_token(logits[-1], temperature=temperature)
            
            # Add to sequence
            tokens = torch.cat([tokens, next_token.unsqueeze(0)])
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