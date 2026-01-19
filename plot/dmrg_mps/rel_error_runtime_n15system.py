import numpy as np
import matplotlib.pyplot as plt

# Relative error data
test_cases = [
    {   'name': r'$\chi_{\max}(\psi)=10$',
        'ranks': [10,20,30,40,50,53,55],
        'rsi_rel_error':[5.53e-2, 5.55e-3, 1.13e-3, 1.17e-4, 2.14e-5, 6.29e-6, 2.07e-15],
        'tci_rel_error':[1.95e-2, 1.12e-3, 8.49e-5, 1.35e-5, 1.58e-6, 4.51e-7, 3.51e-15],
        'dir_rel_error':[1.13e-2, 4.97e-4, 3.57e-5, 4.13e-6, 3.98e-7, 1.37e-7, 8.21e-15]
    },
    {   'name': r'$\chi_{\max}(\psi)=20$',
        'ranks': [10,30,50,100,150,200,210],
        'rsi_rel_error':[3.48e-2, 1.75e-3, 1.50e-4, 9.73e-6, 9.94e-7, 9.91e-9, 3.97e-15],
        'tci_rel_error':[6.36e-3, 1.26e-4, 1.45e-5, 3.01e-7, 1.18e-8, 1.34e-10, 4.18e-15],
        'dir_rel_error':[3.70e-3, 5.10e-5, 4.59e-6, 7.11e-8, 2.33e-9, 1.94e-11, 1.09e-14]
    },
    {   'name': r'$\chi_{\max}(\psi)=30$',
        'ranks': [10,40,100,200,300,400,470],
        'rsi_rel_error':[2.56e-2, 5.29e-4, 2.28e-5, 4.31e-7, 7.02e-9, 1.23e-10, 4.68e-15],
        'tci_rel_error':[7.14e-3, 4.17e-5, 5.46e-7, 5.85e-9, 2.13e-10, 4.96e-12, 1.09e-14],
        'dir_rel_error':[4.00e-3, 1.56e-5, 1.36e-7, 1.18e-9, 3.19e-11, 6.59e-14, 1.32e-15]
    }
]

# Runtime data (ms)
input_rank = [10, 15, 20, 25, 30]
runtime_vs_rank_rsi = [35, 50, 62, 80, 100]
runtime_vs_rank_dir = [59, 396, 840, 1767, 3270]
runtime_vs_rank_tci_1sweep = [1555, 3387, 7199, 11901, 20632]

colors = ['#C6605C', '#597ABA', '#6ECF8B']

# Bond dimension data
bond_dim_10 = [3,9] + [10 for i in range(15)] + [9,3]
bond_dim_20 = [3,9] + [20 for i in range(15)] + [9,3]
bond_dim_30 = [3,9,27] + [30 for i in range(13)] + [27,9,3]
positions = list(range(1,20))

# Create figure with 2 rows
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3, height_ratios=[1.2, 1])

# First row, first subfigure: Bond dimensions
ax_bond = fig.add_subplot(gs[0, 0])
ax_bond.plot(positions, bond_dim_30, marker='s', linestyle='-', linewidth=2, 
             markersize=6, label=r'$\chi_{\max}(\psi)=30$', color='#A3CEF1')
ax_bond.plot(positions, bond_dim_20, marker='^', linestyle='-', linewidth=1.8, 
             markersize=6, label=r'$\chi_{\max}(\psi)=20$', color='#6096BA')
ax_bond.plot(positions, bond_dim_10, marker='o', linestyle='-', linewidth=1.5, 
             markersize=6, label=r'$\chi_{\max}(\psi)=10$', color='#274C77')

ax_bond.set_xlabel(r'Position $i=1,\ldots,19$', fontsize=14)
ax_bond.set_ylabel(r'(a) Bond Dimension $\chi_i$ of Input $\psi$', fontsize=12)
#ax_bond.set_title('Input Maximum Bond Dimensions', fontsize=13, fontweight='bold')
ax_bond.grid(True, alpha=0.3, linestyle=':')
ax_bond.legend(fontsize=10, loc='best')
ax_bond.set_xticks([1,5,10,15,19])

# First row, second subfigure: Runtime (spans two columns)
ax_runtime = fig.add_subplot(gs[0, 1:])

# Plot algorithm runtimes
ax_runtime.semilogy(input_rank, runtime_vs_rank_rsi, 
             marker='o', linestyle='-', color=colors[0], 
             label='RSI', linewidth=2, markersize=8)
ax_runtime.semilogy(input_rank, runtime_vs_rank_tci_1sweep, 
             marker='s', linestyle='-', color=colors[2], 
             label='TCI (1 sweep)', linewidth=2, markersize=8)
ax_runtime.semilogy(input_rank, runtime_vs_rank_dir, 
             marker='^', linestyle='-', color=colors[1], 
             label='direct', linewidth=2, markersize=8)

# Add reference lines for rank^3 and rank^4 scaling
rank_range = np.linspace(input_rank[0], input_rank[-1], 100)
# Scale the reference lines to fit nicely in the plot
scale_rank3 = runtime_vs_rank_rsi[0] / (input_rank[0]**3)
scale_rank4 = runtime_vs_rank_dir[0] / (input_rank[0]**4)

rank3_line = (scale_rank3) * (rank_range ** 3)
rank4_line = (scale_rank4) * (rank_range ** 4)

ax_runtime.semilogy(rank_range, rank3_line, 
                    linestyle='--', color='gray', linewidth=2, 
                    label=r'$\propto$ $\chi^3$', alpha=0.7)
ax_runtime.semilogy(rank_range, rank4_line, 
                    linestyle=':', color='black', linewidth=2, 
                    label=r'$\propto$ $\chi^4$', alpha=0.7)

ax_runtime.set_xlabel(r'Input $\chi_{\max}(\psi)$', fontsize=14)
ax_runtime.set_ylabel('(b) Runtime (ms)', fontsize=14)
#ax_runtime.set_title('Runtime vs Input Rank', fontsize=13, fontweight='bold')
#ax_runtime.set_ylim([20, 1000000])
ax_runtime.grid(True, alpha=0.3, linestyle=':')
ax_runtime.set_xticks(input_rank)
ax_runtime.legend(fontsize=10, loc='best')


# Second row: Three relative error subfigures
markers = ['o', 's', '^']
for i, case in enumerate(test_cases):
    ax = fig.add_subplot(gs[1, i])
    
    ax.semilogy(case['ranks'], case['rsi_rel_error'], 
                marker=markers[0], linestyle='-', color=colors[0], 
                label='RSI', linewidth=2, markersize=8)
    ax.semilogy(case['ranks'], case['tci_rel_error'], 
                marker=markers[1], linestyle='-', color=colors[2], 
                label='TCI', linewidth=2, markersize=8)
    ax.semilogy(case['ranks'], case['dir_rel_error'], 
                marker=markers[2], linestyle='-', color=colors[1], 
                label='direct', linewidth=2, markersize=8)
    
    ax.set_xlabel(r'Output $\chi_{\max}(|\psi|^2)$', fontsize=14)
    if i == 0:
        ax.set_ylabel(r'(c) Relative Error $\epsilon_r$', fontsize=14)
    else:
        ax.set_ylabel(r'$\epsilon_r$', fontsize=14)
        
    ax.set_title(f"Input {case['name']}", fontsize=14)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(fontsize=12, loc='lower left')

    if i == 0:
        ax.set_xticks([10,20,30,40,50,60])
    if i == 1:
        ax.set_xticks([10,50,100,150,220])
    if i == 2:
        ax.set_xticks([10,100,200,300,400,500])

plt.savefig('algorithm_comparison.pdf', dpi=300, bbox_inches='tight')
plt.show()