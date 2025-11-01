import numpy as np
import matplotlib.pyplot as plt

# input f1/f2 tt 20-digit: r_max = 10, rel_error ~1e-13/1e-14/1e-16 vs real f1/f2
# contract_number = 2 sketch_dim = 20 eps = 0

# Define three test cases with 1D lambda functions
test_cases = [
    {
        # input f1/f2 tt 20-digit: r_max = 10, rel_error ~1e-13 vs real f1/f2
        'name': 'x1=0.35, mu=0.65',
        'f1': lambda x: np.exp(-(x-0.35)**2/0.15**2),
        'f2': lambda x: np.exp(-(x-0.65)**2/0.15**2),
        'ranks': [5, 10, 15, 20],
        'rel_error':[4.19e-4, 1.63e-11, 1.90e-12, 1.72e-12]
    },
    {
        # input f1/f2 tt 20-digit: r_max = 10, rel_error ~1e-14 vs real f1/f2
        'name': 'x1=0.25, mu=0.75',
        'f1': lambda x: np.exp(-(x-0.25)**2/0.15**2),
        'f2': lambda x: np.exp(-(x-0.75)**2/0.15**2),
        'ranks': [5, 10, 15, 20],
        'rel_error':[2.40e-4, 3.85e-11, 1.22e-11, 1.23e-11]
    },
    {
        # input f1/f2 tt: r_max = 10, rel_error ~1e-13 vs real f1/f2
        'name': 'x1=0.15, mu=0.85',
        'f1': lambda x: np.exp(-(x-0.15)**2/0.15**2),
        'f2': lambda x: np.exp(-(x-0.85)**2/0.15**2),
        'ranks': [5, 10, 15, 20],
        'rel_error':[1.41e-4, 4.68e-11, 1.01e-11, 9.72e-12]
    }
]
# Create x values for evaluation
x = np.linspace(0, 1, 100)

# Create 3x3 subplot figure
fig, axes = plt.subplots(3, 3, figsize=(15, 8))

for i, test_case in enumerate(test_cases):
    f1 = test_case['f1']
    f2 = test_case['f2']
    
    # Evaluate functions
    F1 = f1(x)
    F2 = f2(x)
    G = F1 * F2
    
    # Generate simulated error data (replace with actual computation)
    # Error decreases as rank increases
    ranks = test_case['ranks']
    errors = test_case['rel_error']
    
    # Column 1: f1 and f2 shapes
    ax1 = axes[i, 0]
    ax1.plot(x, F1, linewidth=2, label='f1', color='blue')
    ax1.plot(x, F2, linewidth=2, label='f2', color='orange')
    ax1.set_xlabel('x')
    ax1.set_ylabel('value')
    ax1.set_title(f'{test_case["name"]}\nf1 and f2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
        
    # Column 2: g = f1 * f2 shape
    ax2 = axes[i, 1]
    ax2.plot(x, G, linewidth=2, color='green')
    ax2.set_xlabel('x')
    ax2.set_ylabel('value')
    ax2.set_title('g = f1 × f2')
    ax2.grid(True, alpha=0.3)

    # Column 3: Error vs rank line plot
    ax3 = axes[i, 2]
    ax3.plot(ranks, errors, 'o-', linewidth=2, markersize=6, color='red')
    ax3.set_xlabel('Rank')
    ax3.set_ylabel('Error')
    ax3.set_title('Approximation Error vs Rank')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

plt.tight_layout()
plt.savefig('gaussian_multiplication.svg', dpi=300)