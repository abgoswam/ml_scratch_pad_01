import time
import torch
import torch.nn as nn
from tqdm import tqdm

assert torch.cuda.is_available(), "CUDA GPU not found."
torch.backends.cudnn.benchmark = True

# Model/data shape
# L, D, N = 8, 8192, 8192  # layers, width, batch size
L, D, N = 2, 1024, 1024  # layers, width, batch size

# FLOP accounting (rule of thumb)
flop_fwd = N * L * 2 * D * D                  # Linear(D,D) per sample ~2*D^2
flop_bwd = 2 * flop_fwd                       # backward ~ 2x forward
flop_step = flop_fwd + flop_bwd               # one training step (fwd+bwd)
H100_TFLOP_PER_SEC = 3.0 # 989.4
t_theoretical = flop_step / (H100_TFLOP_PER_SEC * 1e12)

print(f"flop_step:{flop_step}")
print(f"t_theoretical:{t_theoretical}")

# Model & opt
layers = []
for _ in range(L):
    layers += [nn.Linear(D, D, bias=True), nn.ReLU()]
model = nn.Sequential(*layers).cuda()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

num_steps = 20
print(f"GPU: {torch.cuda.get_device_name(0)}  |  L={L}, D={D}, N={N}")
print("-" * 72)

for step in tqdm(range(num_steps)):
    x = torch.randn(N, D, device="cuda", dtype=torch.float32)

    torch.cuda.synchronize()
    t0 = time.time()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = model(x)
        loss = ((x - y) ** 2).sum()

    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    t_actual = time.time() - t0

    achieved_tflops = flop_step / t_actual / 1e12
    hfu = 100.0 * achieved_tflops / H100_TFLOP_PER_SEC
    mfu = t_theoretical / t_actual  # ratio of theoretical step time to measured

    print(
        f"Step {step+1:02d}: "
        f"MFU: {100*mfu:6.2f}%  |  "
        f"TFLOP/s: {achieved_tflops:8.2f}  |  "
        f"HFU: {hfu:6.2f}%  |  "
        f"time: {t_actual:6.3f}s"
    )
