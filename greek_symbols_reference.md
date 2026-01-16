# Greek Letters & Symbols Reference for ML/AI

## Quick Reference Table

| # | Symbol | Name | Common Meaning in ML/Stats |
|---|--------|------|---------------------------|
| 1 | **μ** | **mu** | **Mean/Expected value** |
| 2 | **σ** | **sigma** | **Standard deviation** |
| 2 | **σ²** | **sigma-squared** | **Variance** |
| 3 | **Σ** | **Sigma** | **Summation operator** |
| 4 | **α** | **alpha** | **Learning rate** |
| 5 | **θ** | **theta** | **Model parameters** |
| 6 | **ε** | **epsilon** | **Small constant, error** |
| 7 | **λ** | **lambda** | **Regularization parameter** |
| 8 | **γ** | **gamma** | **Discount factor (RL)** |
| 9 | **η** | **eta** | **Learning rate** |
| 10 | **β** | **beta** | **Beta parameters (Adam)** |
| 11 | **τ** | **tau** | **Temperature, time constant** |
| 12 | **δ** | **delta** | **Gradient update, change** |
| 13 | **ρ** | **rho** | **Correlation, momentum** |
| 14 | **π** | **pi** | **Policy (RL), probability** |
| 15 | **φ** | **phi** | **Features, activations** |
| 16 | ω | omega | Weights, angular frequency |
| 17 | Ω | Omega | Sample space, complexity bound |
| 18 | ν | nu | Degrees of freedom |
| 19 | ξ | xi | Random variable |
| 20 | χ | chi | Chi-squared distribution |
| 21 | ψ | psi | Wave function, state |
| 22 | ζ | zeta | Damping ratio |

---

## Detailed Examples

### 1. μ (mu) - Mean

**Memory aid**: μ sounds like "**m**ean"

**Formula**:
$$\mu = \frac{1}{n}\sum_{i=1}^{n} x_i$$

**Example**:
```python
import numpy as np

data = [2, 4, 6, 8, 10]
μ = np.mean(data)  # μ = 6.0
```

---

### 2. σ (sigma) - Standard Deviation

**Memory aid**: σ starts with "**s**" like "**s**tandard deviation"

**Formula**:
$$\sigma = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2}$$

**Variance**:
$$\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2$$

**Example**:
```python
import numpy as np

data = [2, 4, 6, 8, 10]
μ = np.mean(data)      # μ = 6.0
σ = np.std(data)       # σ = 2.83
σ² = np.var(data)      # σ² = 8.0
```

---

### 3. Σ (Sigma) - Summation

**Formula**:
$$\sum_{i=1}^{n} x_i = x_1 + x_2 + ... + x_n$$

**Example**:
```python
# Sum of squares: Σx²
x = [1, 2, 3, 4, 5]
sum_of_squares = sum(xi**2 for xi in x)  # Σx² = 55
```

---

### 4. α (alpha) - Learning Rate

**Used in**: Gradient descent, optimization

**Formula (Gradient Descent)**:
$$\theta_{new} = \theta_{old} - \alpha \cdot \nabla L(\theta)$$

**Example**:
```python
# Gradient descent update
θ = 10.0          # current parameter
∇L = 2.5          # gradient
α = 0.01          # learning rate

θ_new = θ - α * ∇L  # θ_new = 10.0 - 0.025 = 9.975
```

**Common values**: 0.001, 0.01, 0.1

---

### 5. θ (theta) - Model Parameters

**Used in**: Neural networks, regression

**Example**:
```python
import torch
import torch.nn as nn

# Linear model: y = θ₀ + θ₁x
# where θ = [θ₀, θ₁] are the parameters

model = nn.Linear(10, 1)
θ = list(model.parameters())  # Get all θ parameters
```

---

### 6. ε (epsilon) - Small Constant / Error

**Used in**: Numerical stability, error terms

**Example 1 - Numerical Stability**:
```python
import torch

# Avoid division by zero in normalization
x = torch.randn(100)
ε = 1e-8

# Standard normalization with epsilon
x_norm = (x - x.mean()) / (x.std() + ε)
```

**Example 2 - Error Term in Regression**:
$$y = \beta_0 + \beta_1 x + \epsilon$$

where ε represents random error/noise.

---

### 7. λ (lambda) - Regularization

**Used in**: L1, L2 regularization, ridge/lasso regression

**L2 Regularization (Ridge)**:
$$L(\theta) = \text{MSE}(\theta) + \lambda \sum_{i} \theta_i^2$$

**Example**:
```python
import torch
import torch.nn as nn

model = nn.Linear(10, 1)
λ = 0.01  # regularization strength

# Loss with L2 regularization
mse_loss = criterion(predictions, targets)
l2_reg = λ * sum(p.pow(2).sum() for p in model.parameters())
total_loss = mse_loss + l2_reg
```

**Common values**: 0.0001, 0.001, 0.01

---

### 8. γ (gamma) - Discount Factor

**Used in**: Reinforcement Learning

**Bellman Equation**:
$$V(s) = R(s) + \gamma \max_{a} V(s')$$

**Example**:
```python
# Q-learning update
γ = 0.99  # discount factor (future reward weight)
reward = 10
next_q_value = 15

current_q = reward + γ * next_q_value  # 10 + 0.99 * 15 = 24.85
```

**Common values**: 0.9, 0.95, 0.99, 0.999

---

### 9. η (eta) - Learning Rate (Alternative)

Same as α, commonly used in papers as learning rate notation.

**Example**:
```python
# SGD with learning rate η
for epoch in range(epochs):
    for batch in dataloader:
        loss = compute_loss(batch)
        loss.backward()
        
        # Update: θ = θ - η * ∇θ
        optimizer.step()  # Uses η internally
```

---

### 10. β (beta) - Beta Parameters

**Used in**: Adam optimizer, momentum

**Adam Optimizer**:
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$

**Example**:
```python
import torch.optim as optim

# Adam optimizer with β parameters
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999)  # β₁=0.9, β₂=0.999
)
```

**Default values**: β₁ = 0.9, β₂ = 0.999

---

### 11. τ (tau) - Temperature

**Used in**: Softmax temperature, soft updates

**Temperature Softmax**:
$$\text{softmax}(x_i) = \frac{e^{x_i/\tau}}{\sum_j e^{x_j/\tau}}$$

**Example**:
```python
import torch
import torch.nn.functional as F

logits = torch.tensor([2.0, 1.0, 0.1])
τ = 0.5  # temperature

# Higher τ → softer distribution
# Lower τ → sharper distribution
probs = F.softmax(logits / τ, dim=0)
```

**Common values**: 0.1 (sharp), 1.0 (normal), 2.0 (soft)

---

### 12. δ (delta) - Change/Gradient

**Used in**: Gradient updates, delta rule

**Example**:
```python
# Weight update with delta
w_old = 0.5
δ = 0.02  # change/gradient

w_new = w_old + δ  # w_new = 0.52
```

---

### 13. ρ (rho) - Correlation / Momentum

**Pearson Correlation**:
$$\rho_{X,Y} = \frac{\text{cov}(X,Y)}{\sigma_X \sigma_Y}$$

**Example**:
```python
import numpy as np

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

ρ = np.corrcoef(x, y)[0, 1]  # ρ = 1.0 (perfect correlation)
```

---

### 14. π (pi) - Policy in RL

**Used in**: Reinforcement Learning

**Policy**:
$$\pi(a|s) = P(\text{action } a | \text{state } s)$$

**Example**:
```python
# Policy π selects action given state
def π(state):
    """Policy function"""
    if state == "hungry":
        return "eat"
    else:
        return "sleep"
```

---

## Z-Score Normalization (Using μ and σ)

**Formula**:
$$z = \frac{x - \mu}{\sigma}$$

**Complete Example**:
```python
import numpy as np

# Original data
data = np.array([2, 4, 6, 8, 10])

# Calculate μ and σ
μ = np.mean(data)  # μ = 6.0
σ = np.std(data)   # σ = 2.83

# Z-score normalization
z_scores = (data - μ) / σ
# z_scores = [-1.41, -0.71, 0.0, 0.71, 1.41]

# Verify: mean ≈ 0, std ≈ 1
print(f"Mean: {z_scores.mean():.2f}")  # ≈ 0.0
print(f"Std:  {z_scores.std():.2f}")   # ≈ 1.0
```

---

### 15. φ (phi) - Feature Map / Basis Functions

**Used in**: Feature transformations, kernel methods, activation functions

**Formula (Feature Mapping)**:
$$\phi: \mathbb{R}^n \rightarrow \mathbb{R}^m$$

**Example 1 - Feature Transformation**:
```python
import torch
import torch.nn as nn

# φ as activation function
def φ(x):
    """Feature transformation/activation"""
    return torch.relu(x)

x = torch.randn(10, 5)
features = φ(x)  # Apply feature map
```

**Example 2 - Kernel Methods**:
```python
# Polynomial feature map
def φ_poly(x):
    """Map x to polynomial features: [x, x²]"""
    return torch.stack([x, x**2], dim=-1)

x = torch.tensor([1.0, 2.0, 3.0])
φ_x = φ_poly(x)  # [[1, 1], [2, 4], [3, 9]]
```

---

### 16. ω (omega) - Weights / Angular Frequency

**Used in**: Neural network weights, signal processing

**Example 1 - Network Weights**:
```python
import torch.nn as nn

# ω typically represents weights in neural networks
linear = nn.Linear(10, 5)
ω = linear.weight  # Weight matrix (5, 10)
b = linear.bias    # Bias vector (5,)

# Forward pass: y = ωx + b
```

**Example 2 - Angular Frequency**:
```python
import numpy as np
import matplotlib.pyplot as plt

# Signal: x(t) = A*sin(ωt)
A = 1.0       # amplitude
ω = 2 * np.pi  # angular frequency (1 Hz)
t = np.linspace(0, 2, 1000)

signal = A * np.sin(ω * t)
```

---

### 17. Ω (Omega) - Sample Space / Complexity Bound

**Used in**: Probability theory, big-O notation

**Example 1 - Sample Space in Probability**:
```python
import random

# Sample space Ω for dice roll
Ω = {1, 2, 3, 4, 5, 6}

# Event: rolling even number
A = {2, 4, 6}

# Probability P(A) = |A| / |Ω|
P_A = len(A) / len(Ω)  # 0.5
```

**Example 2 - Big-Omega Notation**:
```python
# Ω(n) means "at least order n"
# Algorithm has lower bound of Ω(n log n)

def merge_sort(arr):
    """
    Time complexity: Ω(n log n)
    - Best case: Ω(n log n)
    - Worst case: O(n log n)
    """
    if len(arr) <= 1:
        return arr
    # ... merge sort implementation
```

---

### 18. ν (nu) - Degrees of Freedom

**Used in**: Statistical tests, t-distribution, chi-squared

**Formula (t-distribution)**:
$$t = \frac{\bar{x} - \mu}{s / \sqrt{n}}$$ 
with ν = n - 1 degrees of freedom

**Example**:
```python
import scipy.stats as stats
import numpy as np

# Sample data
data = [2.3, 2.5, 2.7, 2.4, 2.6]
n = len(data)
ν = n - 1  # degrees of freedom = 4

# t-test
x_bar = np.mean(data)
s = np.std(data, ddof=1)
μ_0 = 2.0  # hypothesized mean

t_stat = (x_bar - μ_0) / (s / np.sqrt(n))
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=ν))

print(f"Degrees of freedom: ν = {ν}")
print(f"t-statistic: {t_stat:.3f}")
```

---

### 19. ξ (xi) - Random Variable

**Used in**: Probability theory, stochastic processes

**Example 1 - Random Variable**:
```python
import numpy as np

# ξ represents a random variable
# ξ ~ N(0, 1): normally distributed

ξ = np.random.randn(1000)  # Sample from standard normal

# Properties
print(f"E[ξ] = {ξ.mean():.3f}")    # Expected value ≈ 0
print(f"Var[ξ] = {ξ.var():.3f}")   # Variance ≈ 1
```

**Example 2 - Stochastic Gradient Descent**:
```python
# Mini-batch gradient with noise
# ∇L_batch = ∇L + ξ
# where ξ is noise from sampling

import torch

def sgd_with_noise(gradient, ξ):
    """
    Stochastic gradient = true gradient + noise
    """
    return gradient + ξ

true_grad = torch.tensor([1.0, 2.0, 3.0])
ξ = torch.randn(3) * 0.1  # Random noise
noisy_grad = sgd_with_noise(true_grad, ξ)
```

---

### 20. χ (chi) - Chi-Squared Distribution

**Used in**: Statistical tests, variance testing

**Chi-Squared Test Statistic**:
$$\chi^2 = \sum_{i=1}^{n} \frac{(O_i - E_i)^2}{E_i}$$

where O = observed, E = expected

**Example**:
```python
import numpy as np
from scipy.stats import chi2

# Chi-squared goodness of fit test
observed = np.array([40, 30, 30])
expected = np.array([33.33, 33.33, 33.33])

# Calculate χ² statistic
χ² = np.sum((observed - expected)**2 / expected)

# Degrees of freedom
df = len(observed) - 1  # ν = 2

# p-value
p_value = 1 - chi2.cdf(χ², df)

print(f"χ² = {χ²:.3f}")
print(f"df = {df}")
print(f"p-value = {p_value:.3f}")
```

---

### 21. ψ (psi) - Wave Function / State

**Used in**: Quantum computing, state representation

**Example 1 - Quantum State**:
```python
import numpy as np

# Quantum state ψ (qubit)
# |ψ⟩ = α|0⟩ + β|1⟩

α = 1/np.sqrt(2)
β = 1/np.sqrt(2)

ψ = np.array([α, β])  # State vector

# Probability of measuring |0⟩
P_0 = abs(α)**2  # 0.5

# Probability of measuring |1⟩
P_1 = abs(β)**2  # 0.5

# Normalization: |α|² + |β|² = 1
print(f"Normalized: {P_0 + P_1}")  # 1.0
```

**Example 2 - State in RL**:
```python
# ψ can represent state in some contexts
class State:
    def __init__(self, position, velocity):
        self.ψ = {
            'position': position,
            'velocity': velocity
        }
    
# Current state
ψ_t = State(position=10, velocity=2)
```

---

### 22. ζ (zeta) - Damping Ratio

**Used in**: Control theory, oscillating systems

**Second-Order System**:
$$\ddot{x} + 2\zeta\omega_n\dot{x} + \omega_n^2 x = 0$$

where ζ = damping ratio

**Example**:
```python
import numpy as np
import matplotlib.pyplot as plt

def second_order_response(t, ζ, ω_n):
    """
    Step response of second-order system
    ζ < 1: underdamped (oscillations)
    ζ = 1: critically damped
    ζ > 1: overdamped
    """
    if ζ < 1:
        # Underdamped
        ω_d = ω_n * np.sqrt(1 - ζ**2)
        y = 1 - np.exp(-ζ*ω_n*t) * (np.cos(ω_d*t) + 
                                      (ζ/np.sqrt(1-ζ**2))*np.sin(ω_d*t))
    elif ζ == 1:
        # Critically damped
        y = 1 - np.exp(-ω_n*t) * (1 + ω_n*t)
    else:
        # Overdamped
        s1 = -ζ*ω_n + ω_n*np.sqrt(ζ**2 - 1)
        s2 = -ζ*ω_n - ω_n*np.sqrt(ζ**2 - 1)
        y = 1 + (s2*np.exp(s1*t) - s1*np.exp(s2*t))/(s2-s1)
    return y

t = np.linspace(0, 5, 1000)
ω_n = 2.0  # Natural frequency

# Different damping ratios
ζ_values = [0.2, 0.5, 1.0, 2.0]
for ζ in ζ_values:
    y = second_order_response(t, ζ, ω_n)
    plt.plot(t, y, label=f'ζ = {ζ}')

plt.legend()
plt.title('Effect of Damping Ratio ζ')
plt.xlabel('Time')
plt.ylabel('Response')
```

---

## Memory Aids Summary

| Symbol | Memory Trick |
|--------|--------------|
| μ | **mu** → **m**ean |
| σ | **s**igma → **s**tandard deviation |
| α | **a**lpha → learning r**a**te |
| θ | **th**eta → **th**e parameters |
| ε | **e**psilon → **e**rror/small |
| λ | **l**ambda → **l**oss regularization |
| γ | **g**amma → future rewards (**g**o forward) |
| β | **b**eta → Adam **b**etas |
| τ | **t**au → **t**emperature |

---

## Common Formulas Reference

### Normal Distribution (Gaussian)
$$p(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}$$

Where:
- μ = mean
- σ = standard deviation
- π ≈ 3.14159

### Gradient Descent
$$\theta_{t+1} = \theta_t - \alpha \nabla_\theta L(\theta_t)$$

Where:
- θ = parameters
- α = learning rate
- ∇ = gradient
- L = loss function

### L2 Regularization
$$L_{total} = L_{data} + \lambda \sum_{i} \theta_i^2$$

Where:
- λ = regularization strength
- θ = parameters

### Adam Optimizer
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\theta_t = \theta_{t-1} - \alpha \frac{m_t}{\sqrt{v_t} + \epsilon}$$

Where:
- β₁, β₂ = momentum parameters
- α = learning rate
- ε = small constant for numerical stability
