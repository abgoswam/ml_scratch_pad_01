import torch
import torch.nn as nn

class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)        # (n, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)    # (n, d_out)
        
        attn_scores = queries @ keys.T  # (n, n)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1) # (n, n)

        context_vec = attn_weights @ values  # (n, d_out)
        return context_vec
    
if __name__ == "__main__":

    torch.manual_seed(123)

    inputs = torch.tensor(
        [[0.43, 0.15, 0.89], # Your     (x^1)
        [0.55, 0.87, 0.66], # journey  (x^2)
        [0.57, 0.85, 0.64], # starts   (x^3)
        [0.22, 0.58, 0.33], # with     (x^4)
        [0.77, 0.25, 0.10], # one      (x^5)
        [0.05, 0.80, 0.55]] # step     (x^6)
    )
    
    d_in = inputs.shape[1] # the input embedding size, d=3
    d_out = 2 # the output embedding size, d=2

    torch.manual_seed(789)
    sa_v2 = SelfAttention_v2(d_in, d_out)
    print(sa_v2(inputs))

    # Reuse the query and key weight matrices of the
    # SelfAttention_v2 object from the previous section for convenience
    queries = sa_v2.W_query(inputs)
    keys = sa_v2.W_key(inputs) 
    attn_scores = queries @ keys.T
    print(f"attn_scores:\n{attn_scores}") # (b, b)

    attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
    print(f"attn_weights:\n{attn_weights}")   # (b, b)

    context_length = attn_scores.shape[0]
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
    print(masked)

    attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
    print(attn_weights)

    torch.manual_seed(123)
    dropout = torch.nn.Dropout(0.5) # dropout rate of 50%
    print(dropout(attn_weights))
    print(dropout)
    print("done")