import matplotlib.pyplot as plt
import numpy as np

# Data 1
input_rank = [10, 20, 30, 40, 50]
runtime_vs_rank_rsi = [29.87, 32.75, 43.31, 47.82, 55.88]
runtime_vs_rank_dir = [43.34, 389.21, 1669.38, 5464.47, 12720.77]
runtime_vs_rank_tci_1sweep = [7409.53, 17272.75, 50869.19, 110388.21, 197684.15]

# Data 2
product_cases = [
    {
        'name': r'$g=f_1f_2$',
        'runtime RSI': 30.92,
        'runtime direct': 43.34,
        'runtime TCI': 9394.12
    },
    {
        'name': r'$g=f_1f_2f_2$',
        'runtime RSI': 37.41,
        'runtime direct': 2170.86,
        'runtime TCI': 10071.64
    },
    {
        'name': r'$g=f_1f_1f_2f_2$',
        'runtime RSI': 40.24,
        'runtime direct': 356170.52,
        'runtime TCI': 11990.32
    }
]

rsi_color = "#C6605C"
tci_color = "#6ECF8B"
dir_color = "#597ABA"

# Figure 1: Runtime vs Bond Dimension (Line Plot)
fig1, ax1 = plt.subplots(figsize=(7, 5))

ax1.plot(input_rank, runtime_vs_rank_rsi, 'o-', linewidth=2, markersize=8, label='RSI', color=rsi_color)
ax1.plot(input_rank, runtime_vs_rank_tci_1sweep, 's-', linewidth=2, markersize=8, label='TCI (1 sweep)',color=tci_color)
ax1.plot(input_rank, runtime_vs_rank_dir, '^-', linewidth=2, markersize=8, label='direct', color=dir_color)

# Add scaling reference lines (chi^3 and chi^4)
chi_ref = np.array(input_rank, dtype=float)
# Scale the reference lines to fit nicely in the plot
#scale_chi3 = runtime_vs_rank_rsi[0] / (chi_ref[0]**3)
scale_chi4 = 0.02+runtime_vs_rank_dir[0]/ (chi_ref[0]**4)
#ax1.plot(chi_ref, scale_chi3 * chi_ref**3, '--', linewidth=2, color='gray', alpha=0.6, label=r'$\chi^3$ scaling')
ax1.plot(chi_ref, scale_chi4 * chi_ref**4, '--', linewidth=1.5, color='gray', alpha=0.6, label=r'$\chi^4$ scaling')

ax1.set_xlabel(r'$\chi_{\max}(f)$', fontsize=16)
ax1.set_ylabel('Runtime (ms)', fontsize=16)
ax1.set_xticks(input_rank)
#ax1.set_title('Runtime vs Bond Dimension', fontsize=16)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')
ax1.tick_params(labelsize=14)

plt.tight_layout()
plt.savefig('runtime_vs_bond_dimension.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Figure 2: Runtime for Different Product Cases (Bar Plot)
fig2, ax2 = plt.subplots(figsize=(5, 5))

# Extract data for bar plot
product_names = [case['name'] for case in product_cases]
runtime_rsi = [case['runtime RSI'] for case in product_cases]
runtime_direct = [case['runtime direct'] for case in product_cases]
runtime_tci = [case['runtime TCI'] for case in product_cases]

# Set up bar positions
x = np.arange(len(product_names))
width = 0.15

# Create bars
bars1 = ax2.bar(x - width, runtime_rsi, width, label='RSI', alpha=0.8, color=rsi_color)
bars2 = ax2.bar(x, runtime_direct, width, label='direct', alpha=0.8, color=dir_color)
bars3 = ax2.bar(x + width, runtime_tci, width, label='TCI', alpha=0.8, color=tci_color)

# Customize plot
ax2.set_xlabel('Product Case', fontsize=16)
ax2.set_ylabel('Runtime (ms)', fontsize=16)
#ax2.set_title('Runtime Comparison for Different Product Cases', fontsize=16)
ax2.set_xticks(x)
ax2.set_xticklabels(product_names, fontsize=10)
ax2.legend(fontsize=12)
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, axis='y')
ax2.tick_params(labelsize=16)

plt.tight_layout()
plt.savefig('runtime_product_cases.pdf', dpi=300, bbox_inches='tight')
plt.show()

