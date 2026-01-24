import matplotlib.pyplot as plt
import numpy as np

# Data 2
product_cases = [
    {
        'name': r'$g=f_1f_2$',
        'runtime RSI': 31,
        'runtime direct': 43,
        'runtime TCI': 9394
    },
    {
        'name': r'$g=f_1f_2f_2$',
        'runtime RSI': 37,
        'runtime direct': 2170,
        'runtime TCI': 10071
    },
    {
        'name': r'$g=f_1f_1f_2f_2$',
        'runtime RSI': 40,
        'runtime direct': 356170,
        'runtime TCI': 11990
    }
]

rsi_color = "#C6605C"
tci_color = "#6ECF8B"
dir_color = "#597ABA"

fig2, ax2 = plt.subplots(figsize=(6, 4))

# Extract data for bar plot
product_names = [case['name'] for case in product_cases]
runtime_rsi = [case['runtime RSI'] for case in product_cases]
runtime_direct = [case['runtime direct'] for case in product_cases]
runtime_tci = [case['runtime TCI'] for case in product_cases]

# Set up bar positions
x = np.arange(len(product_names))
width = 0.2

# Create bars
bars1 = ax2.bar(x - width, runtime_rsi, width, label='RSI', alpha=0.8, color=rsi_color)
bars2 = ax2.bar(x, runtime_direct, width, label='direct', alpha=0.8, color=dir_color)
bars3 = ax2.bar(x + width, runtime_tci, width, label='TCI', alpha=0.8, color=tci_color)

# Add values on top of bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom', fontsize=9)

# Customize plot
#ax2.set_xlabel('Product Case', fontsize=16)
ax2.set_ylabel('Runtime (ms)', fontsize=14)
#ax2.set_title('Runtime Comparison for Different Product Cases', fontsize=16)
ax2.set_xticks(x)
ax2.set_xticklabels(product_names, fontsize=12)
ax2.legend(fontsize=12)
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, axis='y')
ax2.tick_params(labelsize=16)

plt.tight_layout()
plt.savefig('runtime_product_cases.pdf', dpi=300, bbox_inches='tight')
