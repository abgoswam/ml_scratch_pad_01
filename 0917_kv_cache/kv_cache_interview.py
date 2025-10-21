"""
Interview Question: Implement KV Cache for Transformer Attention

You are tasked with implementing a KV (Key-Value) cache mechanism to optimize 
inference in transformer models. The KV cache stores previously computed 
key and value vectors to avoid redundant computations during autoregressive generation.

Your task:
1. Implement the KVCache class with proper initialization, update, and retrieval methods
2. Integrate it with the given attention mechanism
3. Ensure the implementation handles batch processing correctly
4. Make sure the cache grows appropriately as new tokens are generated

Boilerplate code is provided below. Fill in the TODOs.
"""

import torch
import torch.nn as nn
import math

class KVCache:
    """
    A Key-Value cache for storing attention keys and values during autoregressive generation.
    """
    
    def __init__(self, max_batch_size: int, max_seq_len: int, n_heads: int, head_dim: int, device: str = "cpu"):
        """
        Initialize the KV cache.
        
        Args:
            max_batch_size: Maximum batch size supported
            max_seq_len: Maximum sequence length supported  
            n_heads: Number of attention heads
            head_dim: Dimension of each head
            device: Device to store tensors on
        """
        # TODO: Initialize cache tensors for keys and values
        # Shape should be [max_batch_size, n_heads, max_seq_len, head_dim]
        self.k_cache = torch.zeros(max_batch_size, n_heads, max_seq_len, head_dim).to(device)
        self.v_cache = torch.zeros(max_batch_size, n_heads, max_seq_len, head_dim).to(device)
        self.running_seq_len = 0
    
    def update(self, keys: torch.Tensor, values: torch.Tensor, start_pos: int) -> None:
        """
        Update the cache with new keys and values.
        
        Args:
            keys: New key vectors [batch_size, n_heads, seq_len, head_dim]
            values: New value vectors [batch_size, n_heads, seq_len, head_dim] 
            start_pos: Position in sequence where these new tokens start
        """
        # TODO: Update the cache with new key/value tensors at the correct positions
        seq_len = keys.shape[2]
        self.k_cache[:, :, start_pos:start_pos + seq_len] = keys
        self.v_cache[:, :, start_pos:start_pos + seq_len] = values
        self.running_seq_len = start_pos + seq_len
    
    def get(self, start_pos: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve keys and values from cache for the given range.
        
        Args:
            start_pos: Starting position in the sequence
            seq_len: Length of sequence to retrieve
            
        Returns:
            Tuple of (keys, values) tensors
        """
        # TODO: Return the appropriate slice of cached keys and values
        keys = self.k_cache[:, :, start_pos:start_pos+seq_len] 
        values = self.v_cache[:, :, start_pos:start_pos+seq_len]
        return keys, values 

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with KV caching support.
    """
    
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        # KV cache - will be initialized when needed
        self.kv_cache = None
        
    def forward(self, x: torch.Tensor, start_pos: int = 0, use_cache: bool = False) -> torch.Tensor:
        """
        Forward pass with optional KV caching.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            start_pos: Starting position for cache (used during generation)
            use_cache: Whether to use KV cache
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        q = self.w_q(x)  # [batch_size, seq_len, d_model]
        k = self.w_k(x)  # [batch_size, seq_len, d_model]  
        v = self.w_v(x)  # [batch_size, seq_len, d_model]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        if use_cache:
            # TODO: Implement caching logic
            # 1. Initialize cache if needed
            # 2. Update cache with current k, v
            # 3. Retrieve full keys/values from cache for attention computation
            # 4. Handle the case where we're generating (start_pos > 0)
            # pass
            if start_pos == 0:
                self.kv_cache = KVCache(batch_size, 
                                        self.max_seq_len, 
                                        self.n_heads, 
                                        self.head_dim,
                                        device=x.device)
            
            self.kv_cache.update(keys=k, values=v, start_pos=start_pos)
            k, v = self.kv_cache.get(start_pos=0, seq_len=self.kv_cache.running_seq_len)
        
        # TODO: Compute attention scores and apply attention
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim) # (bsz, n_heads, seq_len, seq_len)
        # 
        # Apply causal mask if needed (for autoregressive generation)
        if not use_cache:
            mask = torch.tril(torch.ones(scores.size(-2), scores.size(-1), device=x.device))
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        out = attn_weights @ v
        
        # TODO: Reshape output and apply output projection
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(out)


def test_kv_cache():
    """
    Test function to verify your implementation.
    """
    # Test parameters
    batch_size = 2
    seq_len = 4
    d_model = 64
    n_heads = 8
    
    # Create model and input
    model = MultiHeadAttention(d_model, n_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    
    print("Testing KV Cache Implementation...")
    
    # Test 1: Forward pass without cache
    print("Test 1: Forward pass without cache")
    out_no_cache = model(x, use_cache=False)
    print(f"Output shape: {out_no_cache.shape}")
    
    # Test 2: Forward pass with cache (first call)
    print("\nTest 2: Forward pass with cache (initialization)")
    out_with_cache = model(x, start_pos=0, use_cache=True)
    print(f"Output shape: {out_with_cache.shape}")
    
    # Test 3: Incremental generation (cache should be used)
    print("\nTest 3: Incremental generation")
    new_token = torch.randn(batch_size, 1, d_model)
    out_incremental = model(new_token, start_pos=seq_len, use_cache=True)
    print(f"Incremental output shape: {out_incremental.shape}")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    test_kv_cache()