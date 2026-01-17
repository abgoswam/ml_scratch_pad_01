# https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture08-nmt.pdf

def beam_search(
        input_seq,
        next_token_logits_fn,         # returns logits V
        max_new_tokens,
        eos_token,
        beam_size
) -> list[int]:        # returns list with highest (prompt+res) values