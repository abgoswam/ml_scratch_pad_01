
import torch
torch.manual_seed(123)
import torch.nn as nn
import matplotlib.pyplot as plt

def plot_samples(x_before, x_after):
    # x shape (b, emb_size)
    # draw a histogram of all the different b rows.  plot each rown in a separate color.
    # display the mean and variance of rach row
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
    
    # Plot before (left subplot)
    for i in range(x_before.shape[0]):
        row = x_before[i].detach().numpy()
        mean_val = row.mean()
        var_val = row.var()
        color = colors[i % len(colors)]
        
        ax1.hist(row, alpha=0.6, color=color, bins=20, 
                label=f'Row {i}: μ={mean_val:.3f}, σ²={var_val:.3f}')
    
    ax1.legend()
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Before Layer Norm')
    
    # Plot after (right subplot)
    for i in range(x_after.shape[0]):
        row = x_after[i].detach().numpy()
        mean_val = row.mean()
        var_val = row.var()
        color = colors[i % len(colors)]
        
        ax2.hist(row, alpha=0.6, color=color, bins=20, 
                label=f'Row {i}: μ={mean_val:.3f}, σ²={var_val:.3f}')
    
    ax2.legend()
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('After Layer Norm')
    
    plt.tight_layout()
    plt.show()

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        # self.scale = nn.Parameter(torch.ones(emb_dim))
        # self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        return norm_x
        # return self.scale * norm_x + self.shift

# Implement a custom layer norm, so the values are in range [0,1]
class CustomLayerNorm(nn.Module):
    def forward(self, x):
        _min, _ = torch.min(x, dim=-1, keepdim=True)
        _max, _ = torch.max(x, dim=-1, keepdim=True)

        print(f"_min\n{_min}")
        print(f"_max\n{_max}")

        # Add your normalization here
        normalized_x = (x - _min) / (_max - _min)

        return normalized_x

if __name__ == "__main__":

    batch_example = torch.randn(2, 4)
    print(f"batch_example:\n{batch_example}")

    mean = batch_example.mean(dim=-1, keepdim=True)
    var = batch_example.var(dim=-1, unbiased=False, keepdim=True)
    print("Mean:\n", mean)
    print("Variance:\n", var)

    # ln = LayerNorm(emb_dim=5)
    # out_ln = ln(batch_example)
    # mean = out_ln.mean(dim=-1, keepdim=True)
    # var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
    # print("Mean:\n", mean)
    # print("Variance:\n", var)

    cn = CustomLayerNorm()
    out_cn = cn(batch_example)
    mean = out_cn.mean(dim=-1, keepdim=True)
    var = out_cn.var(dim=-1, unbiased=False, keepdim=True)
    print("Mean:\n", mean)
    print("Variance:\n", var)

    plot_samples(batch_example, out_cn)
    print("done")