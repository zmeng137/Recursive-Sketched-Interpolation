import numpy as np
import matplotlib.pyplot as plt

color_rsi = "#C6605C"
color_dir = "#597ABA"
color_dir_kronecker = "#8BA3CC"  # Lighter shade for Kronecker part

test_cases = [
    {   'name': r'Input $\chi_{\max}(\psi)=20$',
        'rsi_ranks': [50, 100, 150],
        'rsi_rel_error':[1.71e-3, 8.91e-5, 2.05e-6],
        'rsi_runtime':282,
        'dir_ranks': [50, 100, 150],
        'dir_rel_error':[5.97e-4, 2.33e-6, 1.04e-7],
        'dir_runtime': 3020,
        'dir_kronecker_runtime': 130
    },
    {
        'name': r'Input $\chi_{\max}(\psi)=40$',
        'rsi_ranks': [50, 100, 150],
        'rsi_rel_error':[2.64e-3, 6.23e-5, 8.35e-6],
        'rsi_runtime':818,
        'dir_ranks': [50, 100, 150],
        'dir_rel_error':[2.27e-4, 8.12e-6, 4.83e-7],
        'dir_runtime':41877,
        'dir_kronecker_runtime': 670
    },
    {
        'name': r'Input $\chi_{\max}(\psi)=60$',
        'rsi_ranks': [50, 150, 250],
        'rsi_rel_error':[1.18e-2, 7.80e-4, 9.38e-6],
        'rsi_runtime':1846,
        'dir_ranks': [50, 150, 250],
        'dir_rel_error':[6.14e-3, 1.10e-5, 9.97e-7],
        'dir_runtime':188865,
        'dir_kronecker_runtime': 2868
    },
    {
        'name': r'Input $\chi_{\max}(\psi)=80$',
        'rsi_ranks': [50, 150, 250],
        'rsi_rel_error':[6.65e-2, 2.00e-3, 9.75e-5],
        'rsi_runtime':3764,
        'dir_ranks': [50, 150, 250],
        'dir_rel_error':[1.44e-2, 1.69e-4, 7.98e-6],
        'dir_runtime':717857,
        'dir_kronecker_runtime': 8666
    },
    {
        'name': r'$\chi_{\max}(\psi)=100$',
        'rsi_ranks': [100,200,300],
        'rsi_rel_error':[3.64e-3, 5.41e-4, 1.88e-5],
        'rsi_runtime':6687,
        'dir_ranks': [100,200,300],
        'dir_rel_error':[5.25e-4, 2.27e-5, 2.81e-6],
        'dir_runtime':1685695,
        'dir_kronecker_runtime': 20322
    }
]

chi_values = [20, 40, 60, 80, 100]

# Extract data
rsi_runtimes = [case['rsi_runtime'] for case in test_cases]
dir_runtimes = [case['dir_runtime'] for case in test_cases]
dir_kronecker_runtimes = [case['dir_kronecker_runtime'] for case in test_cases]
dir_ttrounding_runtimes = [dir_runtimes[i] - dir_kronecker_runtimes[i] for i in range(len(test_cases))]

# ============================================================================
# FIGURE 1: Deviation Z convergence plots
# ============================================================================
fig1, axes = plt.subplots(1, 5, figsize=(16, 3))

for i, (ax, case) in enumerate(zip(axes, test_cases)):
    # Plot RSI
    if case['rsi_ranks']:
        ax.semilogy(case['rsi_ranks'], case['rsi_rel_error'], 
                   'o-', color=color_rsi, linewidth=2, markersize=8, 
                   label='RSI', markeredgewidth=1.5)
    
    # Plot Direct method
    if case['dir_ranks']:
        ax.semilogy(case['dir_ranks'], case['dir_rel_error'], 
                   '^-', color=color_dir, linewidth=2, markersize=8, 
                   label='direct', markeredgewidth=1.5)
    
    ax.set_xlabel(r'Output $\chi_{\max}(|\psi|^2)$', fontsize=15)
    if i == 0:
        ax.set_ylabel(r'Deviation $Z$', fontsize=15)
    ax.set_title(case['name'], fontsize=15)
    ax.set_xticks(case['rsi_ranks'])
    ax.grid(True, alpha=0.3, linestyle='--')
    
    ax.legend(fontsize=12, framealpha=0.9, loc='upper right')
    ax.set_ylim([case['dir_rel_error'][-1]/10, 1])
    if i < 3:
        ax.set_yticks([1e-7, 1e-5, 1e-3, 1e-1])
    else:
        ax.set_yticks([1e-6, 1e-4, 1e-2, 1])

plt.tight_layout()
plt.savefig("n50_diagonal_deviation.pdf", dpi=300)

# ============================================================================
# FIGURE 2: Runtime analysis
# ============================================================================
fig2 = plt.figure(figsize=(16, 4))
gs = fig2.add_gridspec(1, 3, width_ratios=[1.5, 1.5, 1])
ax1 = fig2.add_subplot(gs[0])
ax2 = fig2.add_subplot(gs[1])
ax3 = fig2.add_subplot(gs[2])

# ========== Subplot 1: Runtime comparison bar plot with reference lines ==========
x = np.arange(len(test_cases))
width = 0.15

bars1 = ax1.bar(x - width/2, rsi_runtimes, width, label='RSI', 
                color=color_rsi, alpha=0.8, edgecolor='black', linewidth=1)

# Stacked bars for Direct method
bars2_kronecker = ax1.bar(x + width/2, dir_kronecker_runtimes, width, label='direct (Kronecker)', 
                          color=color_dir_kronecker, alpha=0.8, edgecolor='black', linewidth=1)
bars2_ttrounding = ax1.bar(x + width/2, dir_ttrounding_runtimes, width, 
                           bottom=dir_kronecker_runtimes, label='direct (TT-rounding)', 
                           color=color_dir, alpha=0.8, edgecolor='black', linewidth=1)

# Add reference lines for chi^3 and chi^4 scaling
chi_range = np.linspace(chi_values[0], chi_values[-1], 100)
x_range = np.linspace(x[0], x[-1], 100)

# Scale reference lines to align with the data
scale_chi3 = rsi_runtimes[0] / (chi_values[0]**2)
scale_chi4 = dir_runtimes[0] / (chi_values[0]**4)

chi3_line = scale_chi3 * (chi_range ** 2)
chi4_line = scale_chi4 * (chi_range ** 4)

#ax1.plot(x_range, chi3_line, ':', color=color_rsi, linewidth=2.5, 
#         label=r'$\propto \chi^2$', alpha=0.7, zorder=2)
#ax1.plot(x_range, chi4_line, ':', color=color_dir, linewidth=2.5, 
#         label=r'$\propto \chi^4$', alpha=0.7, zorder=2)

ax1.set_xlabel(r'Input $\chi_{\max}(\psi)$', fontsize=16)
ax1.set_ylabel('(a) Runtime (ms)', fontsize=16)
ax1.set_yscale('log')
ax1.set_ylim([1e2, 1e7])
ax1.set_xticks(x)
ax1.set_xticklabels(['20', '40', '60', '80', '100'], fontsize=13)
ax1.legend(fontsize=12, framealpha=0.9, loc='upper left')
ax1.grid(True, alpha=0.3, axis='y', linestyle='--', which='both')

# Add value labels on bars
def autolabel(bars, ax, x_offset=-7):
    for bar in bars:
        height = bar.get_height()
        if height > 0 and not np.isnan(height):
            if height >= 10000:
                label = f'{height:.2e}'
            else:
                label = f'{height:.0f}'
            ax.annotate(label,
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(x_offset, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)

autolabel(bars1, ax1, x_offset=-9)
# For stacked bars, show total height on top
for i, bar in enumerate(bars2_ttrounding):
    height = bar.get_height() + dir_kronecker_runtimes[i]
    if height > 0 and not np.isnan(height):
        if height >= 10000:
            label = f'{height:.2e}'
        else:
            label = f'{height:.0f}'
        ax1.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(-9, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

# ========== Subplot 2: Line plot comparing RSI and Kronecker runtime (linear scale) ==========
ax2.plot(chi_values, rsi_runtimes, 'o-', color=color_rsi, linewidth=2.5, 
         markersize=10, markeredgewidth=1.5, label='RSI')
ax2.plot(chi_values, dir_runtimes, 's-', color=color_dir, linewidth=2.5, 
         markersize=10, markeredgewidth=1.5, label='direct (total)')
ax2.plot(chi_values, dir_kronecker_runtimes, '^-', color=color_dir_kronecker, linewidth=2.5, 
         markersize=10, markeredgewidth=1.5, label='direct (Kronecker)')

ax2.set_xlabel(r'Input $\chi_{\max}(\psi)$', fontsize=16)
ax2.set_ylabel('(b) Runtime (ms)', fontsize=16)
ax2.set_ylim([-1000, 23000])
ax2.set_xticks(chi_values)
ax2.set_xticklabels(['20', '40', '60', '80', '100'], fontsize=13)
ax2.legend(fontsize=12, framealpha=0.9, loc='upper center')
ax2.grid(True, alpha=0.3, linestyle='--')


# Add value labels on points
for chi, rsi_time, kron_time in zip(chi_values, rsi_runtimes, dir_kronecker_runtimes):
    # RSI labels below the line
    
    if chi >= 60:  # For larger chi, place label above to avoid overlap
        ax2.annotate(f'{rsi_time:.0f}',
                    xy=(chi, rsi_time),
                    xytext=(0, -8),
                    textcoords="offset points",
                    ha='center', va='top', fontsize=9)
        # Direct Kronecker labels above the line
        ax2.annotate(f'{kron_time:.0f}',
                    xy=(chi, kron_time),
                    xytext=(-5, 8),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

# ========== Subplot 3: Speedup plots ==========
# Calculate speedups
speedup_vs_kronecker = [dir_kronecker_runtimes[i] / rsi_runtimes[i] for i in range(len(test_cases))]
speedup_vs_total = [dir_runtimes[i] / rsi_runtimes[i] for i in range(len(test_cases))]

# Plot speedups
ax3.plot(chi_values, speedup_vs_total, 'o-', color=color_dir, linewidth=2.5, 
         markersize=10, markeredgewidth=1.5, label='vs direct (total)')
ax3.plot(chi_values, speedup_vs_kronecker, 's-', color=color_dir_kronecker, linewidth=2.5, 
         markersize=10, markeredgewidth=1.5, label='vs direct (Kronecker)')


ax3.set_xlabel(r'Input $\chi_{\max}(\psi)$', fontsize=16)
ax3.set_ylabel('(c) Speedup of RSI', fontsize=16)
ax3.set_xticks(chi_values)
ax3.set_ylim([0.3, 500])
ax3.set_yscale('log')
ax3.set_xticklabels(['20', '40', '60', '80', '100'], fontsize=13)
ax3.legend(fontsize=12, framealpha=0.9, loc='center right')
ax3.grid(True, alpha=0.3, linestyle='--')

# Add value labels on points
for chi, speed_kron, speed_total in zip(chi_values, speedup_vs_kronecker, speedup_vs_total):
    ax3.annotate(f'{speed_kron:.1f}×',
                xy=(chi, speed_kron),
                xytext=(0, 8),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax3.annotate(f'{speed_total:.0f}×',
                xy=(chi, speed_total),
                xytext=(0, 8),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig("n50_runtime_analysis.pdf", dpi=300)