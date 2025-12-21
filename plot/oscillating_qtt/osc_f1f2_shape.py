import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

f1 = lambda x, B_const: np.cos(x * (2 ** B_const)) * np.exp(- x * x) + 4 * np.exp(x) - 3 * x * x + 10 * x + 1
f2 = lambda x, B_const: np.sin(x * (2 ** B_const)) * (np.exp(x * x) + 5 * x + 2) - 4*x

B = 10

f1_color = "#C64462"
f2_color = "#4E96CD"

interval_list = (0.5, 0.55)
x_ranges = [(0, 0.5), (0.5, 0.55), (0.55, 1)]

n_points = 2**18

width_ratios = [2, 3, 2]

fig = plt.figure(figsize=(10, 6))
gs = GridSpec(1, 3, figure=fig, width_ratios=width_ratios)

axes = []
for p_idx in range(3):
    axes.append(fig.add_subplot(gs[0, p_idx]))

# Calculate global y-limits
y_min, y_max = float('inf'), float('-inf')

for p_idx, (x_start, x_end) in enumerate(x_ranges):
    x = np.linspace(x_start, x_end, n_points)
    y1, y2 = f1(x, B), f2(x, B)
    y_min = min(y_min, y1.min(), y2.min())
    y_max = max(y_max, y1.max(), y2.max())

y_margin = (y_max - y_min) * 0.05
y_min, y_max = y_min - y_margin, y_max + y_margin

# Plot each panel
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
    elif p_idx == 1:
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(left=False, right=False, labelleft=False)
        ax.set_xlabel('x', fontsize=20, labelpad=-10)
        ax.set_xticks(interval_list)
        ax.ticklabel_format(axis='x', style='plain', useOffset=False)
    else:
        ax.spines['left'].set_visible(False)
        ax.tick_params(left=False, labelleft=False)
        ax.set_xticks([0.7, 1])

    if p_idx == 2:
        # Add annotation
        #chi_f1, chi_f2 = 20, 20
        #err_f1, err_f2 = -14, -13
        #text = (f"$\\chi_{{\\max}}(f_1^{{TCI}})={chi_f1}$, $\\epsilon_r(f_1^{{TCI}}) \\sim 10^{{{err_f1}}}$\n"
        #        f"$\\chi_{{\\max}}(f_2^{{TCI}})={chi_f2}$, $\\epsilon_r(f_2^{{TCI}}) \\sim 10^{{{err_f2}}}$")
        #ax.annotate(text, xy=(0.6, 0.35), xycoords='axes fraction',
        #            ha='right', va='center', fontsize=14,
        #            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
        pass

    ax.tick_params(axis='x', labelsize=15) 
    ax.tick_params(axis='y', labelsize=14) 
    
    if p_idx == 1:
        ax.legend(loc='lower center', ncol=2, fontsize=15, frameon=False)
    
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

plt.tight_layout()
plt.subplots_adjust(wspace=0.05)
plt.savefig('functions_plot_B10.pdf', dpi=150, bbox_inches='tight')
