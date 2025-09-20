# %% [markdown]
# <table style="width:100%">
# <tr>
# <td style="vertical-align:middle; text-align:left;">
# <font size="2">
# Supplementary code for the <a href="http://mng.bz/orYv">Build a Large Language Model From Scratch</a> book by <a href="https://sebastianraschka.com">Sebastian Raschka</a><br>
# <br>Code repository: <a href="https://github.com/rasbt/LLMs-from-scratch">https://github.com/rasbt/LLMs-from-scratch</a>
# </font>
# </td>
# <td style="vertical-align:middle; text-align:left;">
# <a href="http://mng.bz/orYv"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/cover-small.webp" width="100px"></a>
# </td>
# </tr>
# </table>

# %% [markdown]
# # Understanding PyTorch Buffers

# %% [markdown]
# In essence, PyTorch buffers are tensor attributes associated with a PyTorch module or model similar to parameters, but unlike parameters, buffers are not updated during training.
# 
# Buffers in PyTorch are particularly useful when dealing with GPU computations, as they need to be transferred between devices (like from CPU to GPU) alongside the model's parameters. Unlike parameters, buffers do not require gradient computation, but they still need to be on the correct device to ensure that all computations are performed correctly.
# 
# In chapter 3, we use PyTorch buffers via `self.register_buffer`, which is only briefly explained in the book. Since the concept and purpose are not immediately clear, this code notebook offers a longer explanation with a hands-on example.

# %% [markdown]
# ## An example without buffers

# %% [markdown]
# Suppose we have the following code, which is based on code from chapter 3. This version has been modified to exclude buffers. It implements the causal self-attention mechanism used in LLMs:

# %%
import torch
import torch.nn as nn

class CausalAttentionWithoutBuffers(nn.Module):

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec

# %% [markdown]
# We can initialize and run the module as follows on some example data:

# %%
torch.manual_seed(123)

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

batch = torch.stack((inputs, inputs), dim=0)
context_length = batch.shape[1]
d_in = inputs.shape[1]
d_out = 2

ca_without_buffer = CausalAttentionWithoutBuffers(d_in, d_out, context_length, 0.0)

with torch.no_grad():
    context_vecs = ca_without_buffer(batch)

print(context_vecs)

# %% [markdown]
# So far, everything has worked fine so far.
# 
# However, when training LLMs, we typically use GPUs to accelerate the process. Therefore, let's transfer the `CausalAttentionWithoutBuffers` module onto a GPU device.
# 
# Please note that this operation requires the code to be run in an environment equipped with GPUs.

# %%
print("Machine has GPU:", torch.cuda.is_available())

batch = batch.to("cuda")
ca_without_buffer.to("cuda");

# %% [markdown]
# Now, let's run the code again:

# %%
with torch.no_grad():
    context_vecs = ca_without_buffer(batch)

print(context_vecs)

# %% [markdown]
# Running the code resulted in an error. What happened? It seems like we attempted a matrix multiplication between a tensor on a GPU and a tensor on a CPU. But we moved the module to the GPU!?
# 
# 
# Let's double-check the device locations of some of the tensors:

# %%
print("W_query.device:", ca_without_buffer.W_query.weight.device)
print("mask.device:", ca_without_buffer.mask.device)

# %%
type(ca_without_buffer.mask)

# %% [markdown]
# As we can see, the `mask` was not moved onto the GPU. That's because it's not a PyTorch parameter like the weights (e.g., `W_query.weight`).
# 
# This means we  have to manually move it to the GPU via `.to("cuda")`:

# %%
ca_without_buffer.mask = ca_without_buffer.mask.to("cuda")
print("mask.device:", ca_without_buffer.mask.device)

# %% [markdown]
# Let's try our code again:

# %%
with torch.no_grad():
    context_vecs = ca_without_buffer(batch)

print(context_vecs)

# %% [markdown]
# This time, it worked!
# 
# However, remembering to move individual tensors to the GPU can be tedious. As we will see in the next section, it's easier to use `register_buffer` to register the `mask` as a buffer.

# %% [markdown]
# ## An example with buffers

# %% [markdown]
# Let's now modify the causal attention class to register the causal `mask` as a buffer:

# %%
import torch
import torch.nn as nn

class CausalAttentionWithBuffer(nn.Module):

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # Old:
        # self.mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)

        # New:
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec

# %% [markdown]
# Now, conveniently, if we move the module to the GPU, the mask will be located on the GPU as well:

# %%
ca_with_buffer = CausalAttentionWithBuffer(d_in, d_out, context_length, 0.0)
ca_with_buffer.to("cuda")

print("W_query.device:", ca_with_buffer.W_query.weight.device)
print("mask.device:", ca_with_buffer.mask.device)

# %%
with torch.no_grad():
    context_vecs = ca_with_buffer(batch)

print(context_vecs)

# %% [markdown]
# As we can see above, registering a tensor as a buffer can make our lives a lot easier: We don't have to remember to move tensors to a target device like a GPU manually.

# %% [markdown]
# ## Buffers and `state_dict`

# %% [markdown]
# - Another advantage of PyTorch buffers, over regular tensors, is that they get included in a model's `state_dict`
# - For example, consider the `state_dict` of the causal attention object without buffers

# %%
ca_without_buffer.state_dict()

# %% [markdown]
# - The mask is not included in the `state_dict` above
# - However, the mask *is* included in the `state_dict` below, thanks to registering it as a buffer

# %%
ca_with_buffer.state_dict()

# %% [markdown]
# - A `state_dict` is useful when saving and loading trained PyTorch models, for example
# - In this particular case, saving and loading the `mask` is maybe not super useful, because it remains unchanged during training; so, for demonstration purposes, let's assume it was modified where all `1`'s were changed to `2`'s:

# %%
ca_with_buffer.mask[ca_with_buffer.mask == 1.] = 2.
ca_with_buffer.mask

# %% [markdown]
# - Then, if we save and load the model, we can see that the mask is restored with the modified value

# %%
torch.save(ca_with_buffer.state_dict(), "model.pth")

new_ca_with_buffer = CausalAttentionWithBuffer(d_in, d_out, context_length, 0.0)
new_ca_with_buffer.load_state_dict(torch.load("model.pth"))

new_ca_with_buffer.mask

# %% [markdown]
# - This is not true if we don't use buffers:

# %%
ca_without_buffer.mask[ca_without_buffer.mask == 1.] = 2.

torch.save(ca_without_buffer.state_dict(), "model.pth")

new_ca_without_buffer = CausalAttentionWithoutBuffers(d_in, d_out, context_length, 0.0)
new_ca_without_buffer.load_state_dict(torch.load("model.pth"))

new_ca_without_buffer.mask


