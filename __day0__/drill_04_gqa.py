def grouped_query_attention(Q, K, V, num_kv_heads):
    """
    Q: [B, Hq, T, D]
    K/V: [B, Hkv, T, D]
    num_kv_heads < num_query_heads
    """
    Hq = Q.shape[1]
    Hkv = K.shape[1]
    assert Hq % Hkv == 0
    repeat_factor = Hq // Hkv

    # Expand K, V to match Q heads
    K_expand = K.repeat_interleave(repeat_factor, dim=1)
    V_expand = V.repeat_interleave(repeat_factor, dim=1)

    return self_attention(Q, K_expand, V_expand)

# ---- Example Run ----
B, T, Hq, Hkv, D = 1, 5, 8, 2, 64
Q = torch.randn(B, Hq, T, D)
K = torch.randn(B, Hkv, T, D)
V = torch.randn(B, Hkv, T, D)

out, _ = grouped_query_attention(Q, K, V, num_kv_heads=Hkv)
print(out.shape)  # [1, 8, 5, 64]
