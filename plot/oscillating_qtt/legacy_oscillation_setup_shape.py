import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

f1 = lambda x, B_const: np.cos(x * (2 ** B_const)) * np.exp(- x * x) + 4 * np.exp(x) - 3 * x * x + 10 * x + 1
f2 = lambda x, B_const: np.sin(x * (2 ** B_const)) * (np.exp(x * x) + 5 * x + 2) - 4*x


B_list = [5, 10, 15, 20]

chimax_list = {}
chimax_list[0] = (10,10)
chimax_list[1] = (20,20)
chimax_list[2] = (30,30)
chimax_list[3] = (40,40)

err_list = {}
err_list[0] = (-13,-11)
err_list[1] = (-14,-13)
err_list[2] = (-14,-12)
err_list[3] = (-12,-11)

f1_color = "#9BD1E5"
f2_color = "#5BC480"
g_color = "#D35753"

interval_list = {}
interval_list[0] = (0.5, 0.505)
interval_list[1] = (0.5, 0.5005)
interval_list[2] = (0.5, 0.50005)

x_ranges = {}
x_ranges[0] = [(0, 0.5), (0.5, 0.505), (0.505, 1)]
x_ranges[1] = [(0, 0.5), (0.5, 0.5005), (0.5005, 1)]
x_ranges[2] = [(0, 0.5), (0.5, 0.50005), (0.50005, 1)]

n_points = 2**18

width_ratios = [7] + [2, 3, 2] * 3

fig = plt.figure(figsize=(20, 3))
gs = GridSpec(1, 10, figure=fig, width_ratios=width_ratios)

axes = []
axes.append(fig.add_subplot(gs[0, 0]))
for b_idx in range(3):
    for p_idx in range(3):
        col = 1 + b_idx * 3 + p_idx
        axes.append(fig.add_subplot(gs[0, col]))

# Helper function to create annotation text
def get_annotation_text(b_index):
    chi_f1, chi_f2 = chimax_list[b_index]
    err_f1, err_f2 = err_list[b_index]
    text = (f"$\\chi_{{\\max}}(f_1^{{TCI}})={chi_f1}$, $\\epsilon_r(f_1^{{TCI}}) \\sim 10^{{{err_f1}}}$\n"
            f"$\\chi_{{\\max}}(f_2^{{TCI}})={chi_f2}$, $\\epsilon_r(f_2^{{TCI}}) \\sim 10^{{{err_f2}}}$")
    return text

# Plot B=5 (single axis)
B = 5
x = np.linspace(0, 1, n_points)
y1, y2 = f1(x, B), f2(x, B)
g = y1 * y2

ax = axes[0]
ax.plot(x, y1, label=r"$f_1$", color=f1_color, alpha=0.8)
ax.plot(x, y2, label=r"$f_2$", color=f2_color, alpha=0.8)
#ax.plot(x, g, label=r"$g=f_1 \cdot f_2$", color=g_color, alpha=0.8)
ax.set_xlabel('x', fontsize=16)
ax.set_ylabel('y', fontsize=16)
ax.set_title(f'B = {B}', fontsize=15)
ax.tick_params(axis='x', labelsize=13) 
ax.tick_params(axis='y', labelsize=12) 
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', ncol=3, fontsize=13, frameon=False)

# Add annotation for B=5 (index 0)
ax.annotate(get_annotation_text(0), xy=(0.5, 0.35), xycoords='axes fraction',
            ha='center', va='center', fontsize=14,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

# Plot B=10, 15, 20 (split axes)
for b_idx, B in enumerate(B_list[1:]):
    y_min, y_max = float('inf'), float('-inf')
    
    for p_idx, (x_start, x_end) in enumerate(x_ranges[b_idx]):
        x = np.linspace(x_start, x_end, n_points)
        y1, y2 = f1(x, B), f2(x, B)
        g = y1 * y2
        y_min = min(y_min, y1.min(), y2.min(), g.min())
        y_max = max(y_max, y1.max(), y2.max(), g.max())
    
    y_margin = (y_max - y_min) * 0.05
    y_min, y_max = y_min - y_margin, y_max + y_margin
    
    for p_idx, (x_start, x_end) in enumerate(x_ranges[b_idx]):
        ax_idx = 1 + b_idx * 3 + p_idx
        ax = axes[ax_idx]
        x = np.linspace(x_start, x_end, n_points)
        y1, y2 = f1(x, B), f2(x, B)
        g = y1 * y2
        
        ax.plot(x, y1, label=r"$f_1$", color=f1_color, alpha=0.8)
        ax.plot(x, y2, label=r"$f_2$", color=f2_color, alpha=1)
        #ax.plot(x, g, label=r"$g=f_1 \cdot f_2$", color=g_color, alpha=0.5)

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
            ax.set_title(f'B = {B}', fontsize=17)
            ax.set_xlabel('x', fontsize=16)
            ax.set_xticks(interval_list[b_idx])
            ax.ticklabel_format(axis='x', style='plain', useOffset=False)
            
            # Add annotation in the middle panel (center of the 3-panel group)
            
        else:
            ax.spines['left'].set_visible(False)
            ax.tick_params(left=False, labelleft=False)
            ax.set_xticks([0.7, 1])

        if p_idx == 2:
            ax.annotate(get_annotation_text(b_idx + 1), xy=(0.6, 0.35), xycoords='axes fraction',
                        ha='right', va='center', fontsize=14,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

        ax.tick_params(axis='x', labelsize=13) 
        ax.tick_params(axis='y', labelsize=12) 
        
        if p_idx == 2:
            ax.legend(loc='lower right', ncol=3, fontsize=13, frameon=False)
        
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

for i, ax in enumerate(axes):
    if i == 0:
        continue
    b_idx = (i - 1) // 3 + 1
    pos = ax.get_position()
    ax.set_position([pos.x0 + 0.02 * b_idx, pos.y0, pos.width, pos.height])

plt.savefig('functions_plot.png', dpi=150, bbox_inches='tight')
plt.savefig('functions_plot.pdf', dpi=150, bbox_inches='tight')
plt.show()