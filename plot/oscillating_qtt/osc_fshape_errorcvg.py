import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# Define functions
f1 = lambda x, B_const: np.cos(x * (2 ** B_const)) * np.exp(- x * x) + 4 * np.exp(x) - 3 * x * x + 10 * x + 1
f2 = lambda x, B_const: np.sin(x * (2 ** B_const)) * (np.exp(x * x) + 5 * x + 2) - 4*x

B = 10

# Colors
f1_color = "#C64462"
f2_color = "#4E96CD"
color_rsi = "#C6605C"
color_tci = "#6ECF8B"
color_dir = "#597ABA"

# Function plot parameters
interval_list = (0.5, 0.55)
x_ranges = [(0, 0.5), (0.5, 0.55), (0.55, 1)]
n_points = 2**18
width_ratios = [2, 3, 2]

# Error convergence data
test_cases = [
    {   'name': r'$f_1f_2$',
        'rsi_ranks': [5, 10, 15, 20, 25, 30],
        'rsi_rel_error_p0': [9.51e-1, 1.59e-1, 2.07e-3, 1.71e-5, 6.60e-8, 5.79e-9],
        'rsi_rel_error_p5': [2.53e-1, 8.58e-3, 1.18e-4, 1.31e-5, 6.12e-8, 5.77e-9],
        'rsi_rel_error_p10':[1.42e-1, 5.07e-3, 1.10e-4, 4.14e-6, 2.71e-8, 5.51e-9],

        'tci_ranks': [5, 10, 15, 20],
        'tci_rel_error':[4.19e-2, 1.24e-4, 1.30e-7, 5.45e-9],

        'dir_ranks': [5, 10, 15, 20],
        'dir_rel_error':[1.18e-2, 4.47e-5, 3.09e-8, 4.32e-9],
    }
]

# Create figure with two main subplots
fig = plt.figure(figsize=(17, 5))
main_gs = GridSpec(1, 2, figure=fig, width_ratios=[3, 3], wspace=0.5)

# Left subplot: Function shape with three panels
left_gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=main_gs[0], 
                                   width_ratios=width_ratios, wspace=0.05)

axes = []
for p_idx in range(3):
    axes.append(fig.add_subplot(left_gs[0, p_idx]))

# Calculate global y-limits
y_min, y_max = float('inf'), float('-inf')
for p_idx, (x_start, x_end) in enumerate(x_ranges):
    x = np.linspace(x_start, x_end, n_points)
    y1, y2 = f1(x, B), f2(x, B)
    y_min = min(y_min, y1.min(), y2.min())
    y_max = max(y_max, y1.max(), y2.max())

y_margin = (y_max - y_min) * 0.05
y_min, y_max = y_min - y_margin, y_max + y_margin

# Plot each panel of function shape
for p_idx, (x_start, x_end) in enumerate(x_ranges):
    ax = axes[p_idx]
    x = np.linspace(x_start, x_end, n_points)
    y1, y2 = f1(x, B), f2(x, B)
    
    ax.plot(x, y1, label=r"$f_1$", color=f1_color, alpha=0.8)
    ax.plot(x, y2, label=r"$f_2$", color=f2_color, alpha=1)

    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    
    if p_idx == 0:
        ax.spines['right'].set_visible(False)
        ax.tick_params(right=False)
        ax.set_xticks([0.0, 0.3])
        ax.set_ylabel('(a) Function Value', fontsize=20)
    elif p_idx == 1:
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(left=False, right=False, labelleft=False)
        ax.set_xlabel('x', fontsize=20)
        ax.set_xticks(interval_list)
        ax.ticklabel_format(axis='x', style='plain', useOffset=False)
    else:
        ax.spines['left'].set_visible(False)
        ax.tick_params(left=False, labelleft=False)
        ax.set_xticks([0.7, 1])

    ax.tick_params(axis='x', labelsize=14) 
    ax.tick_params(axis='y', labelsize=14) 
    
    if p_idx == 1:
        ax.legend(loc='lower center', ncol=2, fontsize=16, frameon=False)
    
    # Add broken axis markers
    d = 0.015
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, linewidth=1)
    if p_idx == 0:
        ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    elif p_idx == 1:
        ax.plot((-d, +d), (-d, +d), **kwargs)
        ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    else:
        ax.plot((-d, +d), (-d, +d), **kwargs)
        ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)

# Right subplot: Error convergence
ax_right = fig.add_subplot(main_gs[1])

markers = ['o', 's', '^']
linestyles = ['-', '-.', ':']

for idx, case in enumerate(test_cases):
    # Plot RSI
    ax_right.semilogy(case['rsi_ranks'], case['rsi_rel_error_p0'], 
                marker=markers[0], linestyle=linestyles[2],
                label="RSI (p=0)", linewidth=2, markersize=6, 
                color=color_rsi, alpha=0.7+idx*0.15)
    ax_right.semilogy(case['rsi_ranks'], case['rsi_rel_error_p5'], 
                marker=markers[0], linestyle=linestyles[1],
                label="RSI (p=5)", linewidth=2, markersize=6, 
                color=color_rsi, alpha=0.7+idx*0.15)
    ax_right.semilogy(case['rsi_ranks'], case['rsi_rel_error_p10'], 
                marker=markers[0], linestyle=linestyles[0],
                label="RSI (p=10)", linewidth=2, markersize=6, 
                color=color_rsi, alpha=0.7+idx*0.15)
    
    # Plot TCI
    ax_right.semilogy(case['tci_ranks'], case['tci_rel_error'], 
                marker=markers[1], linestyle=linestyles[0],
                label="TCI", linewidth=2, markersize=6, 
                color=color_tci, alpha=0.7+idx*0.15)
    
    # Plot DIR
    if case['dir_ranks']:
        ax_right.semilogy(case['dir_ranks'], case['dir_rel_error'], 
                    marker=markers[2], linestyle=linestyles[0],
                    label="direct", linewidth=2, markersize=6, 
                    color=color_dir, alpha=0.7+idx*0.15)

ax_right.set_xlabel(r'$\chi_{\max}(g)$', fontsize=20)
ax_right.set_ylabel(r'(b) Relative Error $\epsilon_r$', fontsize=20)
ax_right.set_ylim([1e-9, 1])
ax_right.tick_params(axis='x', labelsize=14)
ax_right.tick_params(axis='y', labelsize=14)
ax_right.grid(True, alpha=0.3)
ax_right.legend(fontsize=14, ncol=1, loc='upper right')

plt.savefig('combined_plot_B10.pdf', dpi=300, bbox_inches='tight')
plt.show()