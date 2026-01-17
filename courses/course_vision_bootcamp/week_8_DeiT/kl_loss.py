import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 1. Fake tiny tokenizer setup
# -----------------------------
# Let's pretend these are the token IDs from real tokenization

# Input (prompt): "Prove that 2 is even."
input_tokens = torch.tensor([[1, 5, 7, 2, 9, 3]])   # shape = [1, 6]

# Teacher output: "Because 2 = 1+1, and 1+1 is divisible by 2, 2 is even."
# (Pretend tokenized)
labels = torch.tensor([[4, 8, 2, 6, 6, 9]])  # shape = [1, 6]

# Student output: "i dont know"
# We DO NOT feed student tokens — the model produces logits from input_tokens only.


# -----------------------------
# 2. Fake tiny student + teacher
# -----------------------------
vocab = 10
embed_dim = 12
seq_len = 6
batch = 1

class TinyModel(nn.Module):
    def __init__(self, vocab, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed_dim)
        self.fc1 = nn.Linear(embed_dim, 32)
        self.fc2 = nn.Linear(32, vocab)

    def forward(self, x):
        e = self.embed(x)                         # [B, T, E]
        h = F.relu(self.fc1(e))                   # [B, T, 32]
        logits = self.fc2(h)                      # [B, T, vocab]
        return logits

teacher = TinyModel(vocab, embed_dim)
student = TinyModel(vocab, embed_dim)


# -----------------------------
# 3. Forward pass
# -----------------------------
teacher_logits = teacher(input_tokens)  # [1, 6, 10]
student_logits = student(input_tokens)  # [1, 6, 10]


# -----------------------------
# 4. Cross entropy on TRUE labels
# -----------------------------
# Flatten for CE
ce_loss = F.cross_entropy(
    student_logits.view(-1, vocab),
    labels.view(-1)
)


# -----------------------------
# 5. KL divergence between teacher + student
# -----------------------------
T = 2.0
alpha = 0.5

log_p_student = F.log_softmax(student_logits / T, dim=-1)   # log probabilities
p_teacher     = F.softmax(teacher_logits / T, dim=-1)       # probabilities

kl_loss = F.kl_div(
    log_p_student,
    p_teacher,
    reduction="batchmean"
) * (T * T)


# -----------------------------
# 6. Combine into final loss
# -----------------------------
loss = (1 - alpha) * ce_loss + alpha * kl_loss


# -----------------------------
# 7. Print results
# -----------------------------
print("Teacher logits:\n", teacher_logits)
print("Student logits:\n", student_logits)
print("\nCE loss: ", ce_loss.item())
print("KL loss: ", kl_loss.item())
print("Final Distillation Loss: ", loss.item())
