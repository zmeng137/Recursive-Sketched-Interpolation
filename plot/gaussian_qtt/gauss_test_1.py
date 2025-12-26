import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.ticker as ticker

# input f1/f2 tt 20-digit: r_max = 10, rel_error ~1e-13/1e-14/1e-16 vs real f1/f2
# contract_number = 2 sketch_dim = 20 eps = 0

sigma = 2 * 0.15 * 0.15

# Define three test cases with 1D lambda functions
test_cases = [
    {   'name': r'$\mu_1=0.4, \mu_2=0.6$',
        'f1': lambda x: np.exp(-(x-0.4)**2/sigma),
        'f2': lambda x: np.exp(-(x-0.6)**2/sigma),
        'ranks': [4, 6, 8, 10, 12],
        'rsi_rel_error':[1.61e-3, 7.69e-6, 3.21e-9,  7.36e-13, 5.56e-15],
        'tci_rel_error':[5.18e-4, 1.51e-6, 2.01e-10, 3.11e-13, 5.50e-15],
        'dir_rel_error':[2.70e-4, 5.41e-7, 8.43e-11, 5.21e-14, 5.21e-15]
    },
    {
        # input f1/f2 tt 20-digit: r_max = 10, rel_error ~1e-14 vs real f1/f2
        'name': r'$\mu_1=0.25, \mu_2=0.75$',
        'f1': lambda x: np.exp(-(x-0.25)**2/sigma),
        'f2': lambda x: np.exp(-(x-0.75)**2/sigma),
        'ranks': [4, 6, 8, 10, 12],
        'rsi_rel_error':[4.03e-4, 2.36e-6, 1.13e-9,  4.26e-13, 8.71e-14],
        'tci_rel_error':[5.18e-4, 1.51e-6, 5.00e-10, 3.75e-13, 3.95e-14],
        'dir_rel_error':[2.70e-4, 5.40e-7, 8.42e-11, 9.81e-14, 2.48e-14]
    },
    {
        # input f1/f2 tt: r_max = 10, rel_error ~1e-13 vs real f1/f2
        'name': r'$\mu_1=0.1, \mu_2=0.9$',
        'f1': lambda x: np.exp(-(x-0.1)**2/sigma),
        'f2': lambda x: np.exp(-(x-0.9)**2/sigma),
        'ranks': [4, 6, 8, 10, 12],
        'rsi_rel_error':[7.01e-4, 2.81e-6, 1.65e-10, 4.89e-12, 2.88e-13],
        'tci_rel_error':[5.17e-4, 1.51e-6, 2.00e-10, 2.38e-12, 9.98e-13],
        'dir_rel_error':[2.70e-4, 5.41e-7, 8.43e-11, 7.38e-13, 5.41e-13]
    }
]
# Create x values for evaluation
x = np.linspace(0, 1, 100)

fig, axes = plt.subplots(2, 3, figsize=(15, 5))

f1_color = "#A4C2A8"
f2_color = "#ACEB98"
g_color = "#C6605C"

rsi_color = "#C6605C"
tci_color = "#6ECF8B"
dir_color = "#597ABA"

for i, test_case in enumerate(test_cases):
    f1 = test_case['f1']
    f2 = test_case['f2']
    
    # Evaluate functions
    F1 = f1(x)
    F2 = f2(x)
    G = F1 * F2

    ranks = test_case['ranks']
    rsi_errors = test_case['rsi_rel_error']
    tci_errors = test_case['tci_rel_error']
    dir_errors = test_case['dir_rel_error']

    ax1 = axes[0,i]
    ax1.plot(x, F1, linewidth=2, label=r'$f_1$', color=f1_color)
    ax1.plot(x, F2, linewidth=2, label=r'$f_2$', color=f2_color)
    ax1.plot(x, G, linewidth=2.5, label=r'$g=f_1 f_2$', color=g_color)
    ax1.set_xlabel('x', fontsize=15,labelpad=-3)
    
    ax1.set_title(f'{test_case["name"]}',fontsize=15, fontweight="bold")
    ax1.legend(loc='right')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1,i]
    ax2.plot(ranks, rsi_errors, 'o-', linewidth=2, markersize=5, color=rsi_color, label="RSI")
    ax2.plot(ranks, tci_errors, 's-', linewidth=2, markersize=5, color=tci_color, label="TCI")
    ax2.plot(ranks, dir_errors, '^-', linewidth=2, markersize=5, color=dir_color, label="direct")

    ax2.set_xlabel(r'$\chi_{\max}(g)$', fontsize=15,labelpad=1)
    #ax2.set_title('Approximation Error vs Rank')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    ax2.legend()

    if i == 0:
        ax1.set_ylabel('(a) Function Value', fontsize=13)
        ax2.set_ylabel(r'(b) Relative Error $\epsilon_r$', fontsize=13)

plt.subplots_adjust(hspace=0.3, wspace=0.4)
plt.savefig('gaussian_multiplication.pdf', dpi=300)
