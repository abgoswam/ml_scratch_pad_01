class TinyLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size, D, H, max_seq_len=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.D = D
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.token_embedding = torch.nn.Embedding(vocab_size, D)
        
        # Positional embeddings
        self.position_embedding = torch.nn.Embedding(max_seq_len, D)
        
        # Attention layer
        self.attention = TinySelfAttention(D, H)
        
        # Language model head
        self.lm_head = torch.nn.Linear(D, vocab_size, bias=False)

    def forward(self, token_ids, position_ids=None, cache=None):
        # token_ids: [T] - sequence of token indices
        T = token_ids.shape[0] if token_ids.dim() > 0 else 1
        
        # If position_ids not provided, create them based on sequence length
        if position_ids is None:
            if cache is None or cache["k"] is None:
                # Prefill: positions are 0, 1, 2, ..., T-1
                position_ids = torch.arange(T, device=token_ids.device)
            else:
                # Generation: position is length of cached sequence
                cached_length = cache["k"].shape[1]  # [H, cached_len, dh]
                position_ids = torch.tensor([cached_length], device=token_ids.device)
        
        # Convert token IDs to embeddings: [T] -> [T, D]
        token_embeds = self.token_embedding(token_ids.unsqueeze(0) if token_ids.dim() == 0 else token_ids)
        
        # Add positional embeddings: [T, D] + [T, D] -> [T, D]
        pos_embeds = self.position_embedding(position_ids)
        x = token_embeds + pos_embeds
        
        # Apply attention: [T, D] -> [T, D]
        x, cache = self.attention(x, cache)
        
        # Convert back to token logits: [T, D] -> [T, vocab_size]
        logits = self.lm_head(x)
        
        return logits, cache
    
def generate_text(model, initial_tokens, max_new_tokens=10, temperature=1.0):
    """Generate text using KV cache with proper positional embeddings"""
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
            # Prefill with positions 0, 1, 2, ..., len(tokens)-2
            logits, cache = model(tokens[:-1], cache=cache)
        
        # Generate new tokens one by one
        for i in range(max_new_tokens):
            # Current position for the new token
            current_position = len(tokens) - 1
            
            # Get logits for the last token at its correct position
            logits, cache = model(
                tokens[-1:], 
                position_ids=torch.tensor([current_position]), 
                cache=cache
            )
            
            # Sample next token
            next_token = sample_next_token(logits[-1], temperature=temperature)
            
            # Add to sequence
            tokens = torch.cat([tokens, next_token.unsqueeze(0)])
            generated_tokens.append(next_token.item())
    
    return tokens, generated_tokens