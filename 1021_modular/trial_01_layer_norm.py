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

if __name__ == "__main__":

    batch_example = torch.randn(2, 5)   
    print(f"batch_example:\n{batch_example}")

    layer = nn.Sequential(
        nn.Linear(5, 128), 
        nn.ReLU()
    )
    
    out = layer(batch_example)
    mean = out.mean(dim=-1, keepdim=True)
    var = out.var(dim=-1, keepdim=True)
    print(f"out:\n{out}")
    print("Mean:\n", mean)
    print("Variance:\n", var)
    
    # working on layer norm.
    out_norm = (out - mean) / torch.sqrt(var)
    mean = out_norm.mean(dim=-1, keepdim=True)
    var = out_norm.var(dim=-1, keepdim=True)
    print("Normalized layer outputs:\n", out_norm)
    print("Mean:\n", mean)
    print("Variance:\n", var)

    plot_samples(out, out_norm)
    print("done")