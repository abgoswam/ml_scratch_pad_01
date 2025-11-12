# MHA

class Mha:
    
    def __init__(self, emb_size, n_heads):
        # pass
        # self.d_in = self.d_out = emb_size
        
        self.head_dim = emb_size // n_heads
        self.q_layer = nn.Linear(emb_size, emb_size)  # fix 
        self.k_layer = nn.Linear(emb_size, emb_size)
        self.v_layer = nn.Linear(emb_size, emb_size)
        
    def forward(self, x):
        
        # pass
        # x  (b, n_tokens, emb_size)
        
        q = self.q_layer(x)  #  (b, n_tokens, emb_size)
        k = self.k_layer(x)  #  (b, n_tokens, emb_size)
        v = self.v_layer(x)  #  (b, n_tokens, emb_size)
        
        q =  q.view(b, n_tokens, n_heads, head_dim).transpose(1, 2) #  (b, n_tokens, n_heads, head_dim) -> (b, n_heads, n_tokens, head_dim)
        k =  q.view(b, n_tokens, n_heads, head_dim).transpose(1, 2) #  (b, n_tokens, n_heads, head_dim) -> (b, n_heads, n_tokens, head_dim)
        v =  q.view(b, n_tokens, n_heads, head_dim).transpose(1, 2) #  (b, n_tokens, n_heads, head_dim) -> (b, n_heads, n_tokens, head_dim)
        
        # 3.
        #    q = (b, n_heads, n_tokens, head_dim)
        #    k = (b, n_heads, n_tokens, head_dim)
        #    v = (b, n_heads, n_tokens, head_dim)
        
        # 4.
        #     attn = q @ k.T(2,3)  -> (b, n_heads, n_tokens, n_tokens)
        #     attn  = softmax(dim=-1) / sqrt(self.head_dim)     # TODO  head_dim or emb_dim ?
        
        attn = q @ k.T(2,3) / sqrt(self.head_dim) # -> (b, n_heads, n_tokens, n_tokens)
        
        attn  = torch.nn.softmax(attn, dim=-1) 
        
        
        # 5
        #     context_vec = attn @  v.T(2,3) ->   (b, n_heads, n_tokens, head_dim)
        #     context_vec   -> (b, n_tokens, n_heads, head_dim)
        #     context_vec   -> (b, n_tokens, emb_size)
        
        