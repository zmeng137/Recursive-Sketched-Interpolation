import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# input f1/f2 tt 20-digit: r_max = 10, rel_error ~1e-13/1e-14/1e-16 vs real f1/f2
# contract_number = 2 sketch_dim = 20 eps = 0

# Define three test cases with 1D lambda functions
test_cases = [
    {   'name': r'$\mu_1=0.45, \mu_2=0.55$',
        'f1': lambda x: np.exp(-(x-0.45)**2/0.15**2),
        'f2': lambda x: np.exp(-(x-0.55)**2/0.15**2),
        'ranks': [5, 10, 15, 20],
        'rel_error':[2.2e-4, 3.05e-12, 1.49e-13, 1.33e-13]
    },
    {
        # input f1/f2 tt 20-digit: r_max = 10, rel_error ~1e-13 vs real f1/f2
        'name': r'$\mu_1=0.35, \mu_2=0.65$',
        'f1': lambda x: np.exp(-(x-0.35)**2/0.15**2),
        'f2': lambda x: np.exp(-(x-0.65)**2/0.15**2),
        'ranks': [5, 10, 15, 20],
        'rel_error':[4.19e-4, 1.63e-11, 1.90e-12, 1.72e-12]
    },
    {
        # input f1/f2 tt 20-digit: r_max = 10, rel_error ~1e-14 vs real f1/f2
        'name': r'$\mu_1=0.25, \mu_2=0.75$',
        'f1': lambda x: np.exp(-(x-0.25)**2/0.15**2),
        'f2': lambda x: np.exp(-(x-0.75)**2/0.15**2),
        'ranks': [5, 10, 15, 20],
        'rel_error':[2.40e-4, 3.85e-11, 1.22e-11, 1.23e-11]
    },
    {
        # input f1/f2 tt: r_max = 10, rel_error ~1e-13 vs real f1/f2
        'name': r'$\mu_1=0.15, \mu_2=0.85$',
        'f1': lambda x: np.exp(-(x-0.15)**2/0.15**2),
        'f2': lambda x: np.exp(-(x-0.85)**2/0.15**2),
        'ranks': [5, 10, 15, 20],
        'rel_error':[1.41e-4, 4.68e-11, 1.01e-11, 9.72e-12]
    }
]
# Create x values for evaluation
x = np.linspace(0, 1, 100)

fig, axes = plt.subplots(2, 4, figsize=(15, 5))

f1_color = "#A4C2A8"
f2_color = "#ACEB98"
g_color = "#CC5551"
approx_color = "#60AA59"

for i, test_case in enumerate(test_cases):
    f1 = test_case['f1']
    f2 = test_case['f2']
    
    # Evaluate functions
    F1 = f1(x)
    F2 = f2(x)
    G = F1 * F2

    ranks = test_case['ranks']
    errors = test_case['rel_error']

    ax1 = axes[0,i]
    ax1.plot(x, F1, linewidth=2.5, label=r'$f_1$', color=f1_color)
    ax1.plot(x, F2, linewidth=2.5, label=r'$f_2$', color=f2_color)
    ax1.plot(x, G, linewidth=3, label=r'$g=f_1 f_2$', color=g_color)
    ax1.set_xlabel('x', fontsize=15)
    
    ax1.set_title(f'{test_case["name"]}',fontsize=15)
    ax1.legend(loc='right')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1,i]
    ax2.plot(ranks, errors, 'o-', linewidth=2, markersize=6, color=approx_color)
    ax2.set_xlabel('Rank', fontsize=15)
    #ax2.set_title('Approximation Error vs Rank')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    axins = inset_axes(ax2, width="50%", height="40%", loc='center right')
    
    axins.plot(x, G, color=g_color, linewidth=1.3)
    axins.plot(x, G, color=approx_color, linewidth=2, linestyle='dotted')
    axins.set_xlim(0, 1)  # Zoom in on x-axis if desired
    axins.set_ylim(0, G.max() * 1.1)  # Zoom in on y-axis to show g clearly
    axins.set_title('Real g vs. \n Approximate g', fontsize=10)
    axins.grid(True, alpha=0.3)
    axins.tick_params(labelsize=8)

    # Arrow
    last_rank = ranks[-1]
    last_error = errors[-1]
    bbox = axins.get_position()
    ax2.annotate('', 
            xy=(0.75, 0.2),  # Point on ax2 in axes fraction (approximate inset left edge)
            xytext=(last_rank, 2* last_error),  # Starting from the last data point
            xycoords='axes fraction', 
            textcoords='data',
            arrowprops=dict(arrowstyle='->', lw=1.5, color="#81A77D", alpha=0.7, linestyle='dashed'))

    if i == 0:
        ax1.set_ylabel('Function Value', fontsize=15)
        ax2.set_ylabel('Relative Error', fontsize=15)

plt.subplots_adjust(hspace=0.3)

plt.savefig('gaussian_multiplication.pdf', dpi=300)
plt.savefig('gaussian_multiplication.svg', dpi=300)