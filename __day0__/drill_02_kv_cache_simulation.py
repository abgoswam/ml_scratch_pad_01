class KVCache:
    def __init__(self, max_len, n_heads, head_dim):
        self.keys = []
        self.values = []
        self.max_len = max_len
        self.n_heads = n_heads
        self.head_dim = head_dim

    def append(self, k_t, v_t):
        """k_t and v_t: [B, H, 1, D]"""
        self.keys.append(k_t)
        self.values.append(v_t)

    def get_stack(self):
        # Return tensors of shape [B, H, T, D]
        return torch.cat(self.keys, dim=2), torch.cat(self.values, dim=2)

# ---- Simulate usage ----
B, H, D = 1, 4, 64
kv = KVCache(max_len=5, n_heads=H, head_dim=D)

for t in range(5):
    k_t = torch.randn(B, H, 1, D)
    v_t = torch.randn(B, H, 1, D)
    kv.append(k_t, v_t)

K_all, V_all = kv.get_stack()
print(K_all.shape)  # [1, 4, 5, 64]
