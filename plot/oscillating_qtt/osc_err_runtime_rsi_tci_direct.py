import numpy as np
import matplotlib.pyplot as plt

color_rsi = "#C6605C"
color_tci = "#6ECF8B"
color_dir = "#597ABA"

test_cases = [
    {   'name': r'$f_1f_2$',
        'rsi_ranks': [5, 10, 15, 20, 25, 30],
        'rsi_rel_error':[3.01e-1, 6.30e-3, 6.04e-5, 7.43e-6, 4.71e-8, 5.13e-9],
        'rsi_runtime':55.12,

        'tci_ranks': [5, 10, 15, 20],
        'tci_rel_error':[4.19e-2, 1.24e-4, 1.30e-7, 5.45e-9],
        'tci_runtime':119238.03,

        'dir_ranks': [5, 10, 15, 20],
        'dir_rel_error':[1.18e-2, 4.47e-5, 3.09e-8, 4.32e-9],
        'dir_runtime':45.57
    },
    {
        'name': r'$f_1f_2^2$',
        'rsi_ranks': [10, 15, 20, 25, 30, 35, 40, 45],
        'rsi_rel_error':[4.00e-2, 5.26e-3, 2.46e-4, 6.18e-5, 9.34e-6, 5.23e-7, 6.46e-8, 5.21e-9],
        'rsi_runtime':65.21,

        'tci_ranks': [10, 15, 20, 25],
        'tci_rel_error':[3.44e-3, 7.2e-05, 3.17e-7, 5.42e-9],
        'tci_runtime':147425.70,

        'dir_ranks': [10, 15, 20, 25],
        'dir_rel_error':[1.13e-3, 2.54e-5, 8.70e-8, 4.28e-9],
        'dir_runtime':2130.12
    },
    {
        'name': r'$f_1^2f_2^2$',
        'rsi_ranks': [10, 20, 30, 40, 50, 60],
        'rsi_rel_error':[4.17e-2, 2.93e-3, 9.50e-5, 7.84e-6, 4.22e-7, 9.84e-9],
        'rsi_runtime':120.12,

        'tci_ranks': [10, 20, 30],
        'tci_rel_error':[2.45E-03, 9.23e-6, 1.10e-8],
        'tci_runtime':283761.13,

        'dir_ranks': [10, 20, 30],
        'dir_rel_error':[9.83e-4, 2.70e-6, 8.53e-9],
        'dir_runtime':422918.02
    }
]

# Figure 1: All relative error plots in one figure
fig1, ax1 = plt.subplots(figsize=(8, 5))

markers = ['o', 's', '^']
linestyles = ['-', '-.', ':']

for idx, case in enumerate(test_cases):
    # Plot RSI
    ax1.semilogy(case['rsi_ranks'], case['rsi_rel_error'], 
                marker=markers[idx], linestyle=linestyles[0],
                label=f"RSI {case['name']}", linewidth=2, markersize=6, color=color_rsi, alpha=0.7+idx*0.15)
    
    # Plot TCI
    ax1.semilogy(case['tci_ranks'], case['tci_rel_error'], 
                marker=markers[idx], linestyle=linestyles[1],
                label=f"TCI {case['name']}", linewidth=2, markersize=6, color=color_tci, alpha=0.7+idx*0.15)
    
    # Plot DIR
    if case['dir_ranks']:
        ax1.semilogy(case['dir_ranks'], case['dir_rel_error'], 
                    marker=markers[idx], linestyle=linestyles[2],
                    label=f"Direct {case['name']}", linewidth=2, markersize=6, color=color_dir, alpha=0.7+idx*0.15)

ax1.set_xlabel(r'$\chi_{\max}(g)$', fontsize=14)
ax1.set_ylabel(r'Relative Error $\epsilon_r$', fontsize=14)
ax1.set_ylim([1e-9, 1])
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10, ncol=3, loc='upper right')
plt.tight_layout()
plt.savefig('error_comparison_all.pdf', dpi=300, bbox_inches='tight')
plt.show()

print("Error plot saved as 'error_comparison_all.pdf'")

# Figure 2: All runtime bars in one figure
fig2, ax2 = plt.subplots(figsize=(8, 5))

width = 0.25
num_cases = len(test_cases)
x = np.arange(num_cases)

# Gather all runtimes
rsi_runtimes = [case['rsi_runtime'] for case in test_cases]
tci_runtimes = [case['tci_runtime'] for case in test_cases]
dir_runtimes = [case['dir_runtime'] if case['dir_runtime'] else 0 for case in test_cases]

# Plot bars
bars1 = ax2.bar(x - width, rsi_runtimes, width, label='RSI', color=color_rsi, alpha=0.8)
bars2 = ax2.bar(x, tci_runtimes, width, label='TCI', color=color_tci, alpha=0.8)
bars3 = ax2.bar(x + width, dir_runtimes, width, label='Direct', color=color_dir, alpha=0.8)

ax2.set_ylabel('Runtime (ms)', fontsize=14)
ax2.set_xlabel('Test Case', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels([case['name'] for case in test_cases], fontsize=14)
ax2.set_yscale('log')
ax2.set_ylim([10, 10**6])
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9, rotation=0)

plt.tight_layout()
plt.savefig('runtime_comparison_all.pdf', dpi=300, bbox_inches='tight')
plt.show()

print("Runtime plot saved as 'runtime_comparison_all.pdf'")