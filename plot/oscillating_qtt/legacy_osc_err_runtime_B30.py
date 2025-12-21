import numpy as np
import matplotlib.pyplot as plt

#B=30
#Input max rank 50. TCI(f1), TCI(f2) ~ 1e-8 error

# Data
target_rank = {}
target_rank['RSI'] = [20,30,40,50,60,70,80]
target_rank['TCI'] = [10,20,30,40,50]
target_rank['DIR+ID'] = [10,20,30,40,50]
target_rank['DIR+SVD'] = [10,20,30,40,50]

rel_err = {}
rel_err['RSI'] = [1.01E-02, 1.71e-3, 2.61e-4, 3.93E-06, 1.69e-7, 3.56e-8, 3.56e-9]
rel_err['TCI'] = [3.33e-02, 2.67E-03, 5.28e-5, 8.85e-07, 1.01e-8]
rel_err['DIR+ID'] = [7.44e-2, 2.40e-3, 1.39e-4, 2.71e-6,2.71e-8]
rel_err['DIR+SVD'] = [2.83e-2, 6.12e-4, 3.01e-5, 3.24e-7, 2.88e-9]


# Create figure
plt.figure(figsize=(10, 6))

# Plot each algorithm
algorithms = ['RSI', 'TCI', 'DIR+ID', 'DIR+SVD']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'D']

for alg, color, marker in zip(algorithms, colors, markers):
    plt.semilogy(target_rank[alg], rel_err[alg], 
                 marker=marker, color=color, linewidth=2, 
                 markersize=8, label=alg)

# Formatting
plt.xlabel('Rank', fontsize=12, fontweight='bold')
plt.ylabel('Relative Error', fontsize=12, fontweight='bold')
plt.title('Error Convergence vs Rank', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, frameon=True, shadow=True)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# Show plot
plt.savefig("err_runtime.png")