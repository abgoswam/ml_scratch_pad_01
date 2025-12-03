#!/usr/bin/env python3
# Demo 7: KV cache — sequential decoding with vs without cache
# Usage:
#   pip install torch
#   python demo7_kv_cache.py
import math, time, torch

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
        # x: [T, D] where T=sequence length, D=model dimension
        T, D = x.shape
        # Q: [T, D] -> [T, D] -> [T, H, Dh] -> [H, T, Dh]
        Q = self.Wq(x).view(T, self.H, self.dh).transpose(0,1)  # [H, T, Dh]

        if cache is None:
            # K,V: [T, D] -> [T, D] -> [T, H, Dh] -> [H, T, Dh]
            K = self.Wk(x).view(T, self.H, self.dh).transpose(0,1)  # [H, T, Dh]
            V = self.Wv(x).view(T, self.H, self.dh).transpose(0,1)  # [H, T, Dh]
        else:
            # Only compute K,V for the new token: x[-1:] is [1, D]
            # k_new, v_new: [1, D] -> [1, D] -> [1, H, Dh] -> [H, 1, Dh]
            k_new = self.Wk(x[-1:]).view(1, self.H, self.dh).transpose(0,1)  # [H, 1, Dh]
            v_new = self.Wv(x[-1:]).view(1, self.H, self.dh).transpose(0,1)  # [H, 1, Dh]
            if cache["k"] is None:
                K, V = k_new, v_new  # [H, 1, Dh] for first token
            else:
                # Concatenate along sequence dimension: [H, t-1, Dh] + [H, 1, Dh] -> [H, t, Dh]
                K = torch.cat([cache["k"], k_new], dim=1)  # [H, t, Dh]
                V = torch.cat([cache["v"], v_new], dim=1)  # [H, t, Dh]
            cache["k"], cache["v"] = K, V

        # Attention scores: Q @ K^T = [H, T, Dh] @ [H, Dh, tK] -> [H, T, tK]
        # where tK is the key sequence length (could be < T with caching)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.dh)  # [H, T, tK]
        tK = scores.size(-1)  # tK = current key sequence length
        
        # Causal mask: prevent attending to future positions [T, tK]
        causal = torch.triu(torch.ones(T, tK), diagonal=1).bool()  # [T, tK]
        scores = scores.masked_fill(causal.unsqueeze(0), float("-inf"))  # [H, T, tK]
        
        # Attention probabilities: [H, T, tK]
        P = torch.softmax(scores, dim=-1)  # [H, T, tK]
        
        # Attention output: P @ V = [H, T, tK] @ [H, tK, Dh] -> [H, T, Dh]
        Y = torch.matmul(P, V)  # [H, T, Dh]
        
        # Reshape back: [H, T, Dh] -> [T, H, Dh] -> [T, D]
        Y = Y.transpose(0,1).contiguous().view(T, D)  # [T, D]
        # Final output projection: [T, D] -> [T, D]
        return self.Wo(Y), cache

def main():
    # Model dimensions: T=sequence_length, D=model_dim, H=num_heads
    T, D, H = 512, 512, 8

    attn = TinySelfAttention(D, H).eval()
    # Input sequence: [T, D] = [512, 512]
    x = torch.randn(T, D)

    t0 = time.time()
    y_full, _ = attn(x, cache=None)
    t1 = time.time()

    t2 = time.time()
    ys_nocache = []
    for t in range(1, T+1):
        # Process sequence of length t: x[:t] is [t, D]
        # y_step is [t, D], we take the last token: [D]
        y_step, _ = attn(x[:t], cache=None)
        ys_nocache.append(y_step[-1])  # [D]
    y_seq_nocache = torch.stack(ys_nocache, dim=0)  # [T, D]
    t3 = time.time()

    t4 = time.time()
    ys_cache = []
    cache = {"k": None, "v": None}
    for t in range(1, T+1):
        # Process sequence of length t: x[:t] is [t, D]
        # With caching, only the new token's K,V are computed
        # y_step is [t, D], we take the last token: [D]
        y_step, cache = attn(x[:t], cache=cache)
        ys_cache.append(y_step[-1])  # [D]
    y_seq_cache = torch.stack(ys_cache, dim=0)  # [T, D]
    t5 = time.time()

    print(f"Full single pass: {(t1-t0):.3f}s")
    print(f"Sequential w/o cache: {(t3-t2):.3f}s")
    print(f"Sequential with cache: {(t5-t4):.3f}s")
    print("Equality (seq nocache vs seq cache) max diff:",
          float((y_seq_nocache - y_seq_cache).abs().max()))

if __name__ == "__main__":
    main()

# === Expected Output (approx) ===
# Full single pass: ~0.1s .. 1.5s        # depends on machine
# Sequential w/o cache: noticeably slower
# Sequential with cache: faster than w/o cache
# Equality (seq nocache vs seq cache) max diff: ~0.0
