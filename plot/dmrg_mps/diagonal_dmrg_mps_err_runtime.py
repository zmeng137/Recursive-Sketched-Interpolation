import numpy as np
import matplotlib.pyplot as plt

color_rsi = "#C6605C"
color_dir = "#597ABA"

test_cases = [
    {   'name': r'Input $\chi_{\max}(\psi)=20$',
        'rsi_ranks': [50, 100, 150],
        'rsi_rel_error':[1.71e-3, 8.91e-5, 2.05e-6],
        'rsi_runtime':2280.12,
        'dir_ranks': [50, 100, 150],
        'dir_rel_error':[5.97e-4, 2.33e-6, 1.04e-7],
        'dir_runtime': 3020.62
    },
    {
        'name': r'Input $\chi_{\max}(\psi)=40$',
        'rsi_ranks': [50, 100, 150],
        'rsi_rel_error':[2.64e-3, 6.23e-5, 8.35e-6],
        'rsi_runtime':4294.66,
        'dir_ranks': [50, 100, 150],
        'dir_rel_error':[2.27e-4, 8.12e-6, 4.83e-7],
        'dir_runtime':35064.73
    },
    {
        'name': r'Input $\chi_{\max}(\psi)=60$',
        'rsi_ranks': [50, 150, 250],
        'rsi_rel_error':[1.18e-2, 7.80e-4, 9.38e-6],
        'rsi_runtime':9549.70,
        'dir_ranks': [50, 150, 250],
        'dir_rel_error':[6.14e-3, 1.10e-5, 9.97e-7],
        'dir_runtime':188865.6
    },
    {
        'name': r'Input $\chi_{\max}(\psi)=80$',
        'rsi_ranks': [50, 150, 250],
        'rsi_rel_error':[6.65e-2, 2.00e-3, 9.75e-5],
        'rsi_runtime':15920.30,
        'dir_ranks': [50, 150, 250],
        'dir_rel_error':[1.44e-2, 1.69e-4, 7.98e-6],
        'dir_runtime':717857.13
    },
    {
        'name': r'$\chi_{\max}(\psi)=100$',
        'rsi_ranks': [100,200,300],
        'rsi_rel_error':[3.64e-3, 5.41e-4, 1.88e-5],
        'rsi_runtime':22647.21,
        'dir_ranks': [100,200,300],
        'dir_rel_error':[5.25e-4, 2.27e-5, 2.81e-6],
        'dir_runtime':1685695.36
    }
]

# Create figure with custom layout
fig = plt.figure(figsize=(16, 6))
gs = fig.add_gridspec(2, 6, height_ratios=[0.6, 1], hspace=0.5, wspace=0.5)

# First row: Error convergence plots
for i, case in enumerate(test_cases):
    ax = fig.add_subplot(gs[0, i])
    
    # Plot RSI
    if case['rsi_ranks']:
        ax.semilogy(case['rsi_ranks'], case['rsi_rel_error'], 
                   'o-', color=color_rsi, linewidth=2, markersize=8, 
                   label='RSI', markeredgewidth=1.5, markeredgecolor='white')
    
    # Plot Direct method
    if case['dir_ranks']:
        ax.semilogy(case['dir_ranks'], case['dir_rel_error'], 
                   '^-', color=color_dir, linewidth=2, markersize=8, 
                   label='Direct', markeredgewidth=1.5, markeredgecolor='white')
    
    ax.set_xlabel(r'Output $\chi_{\max}(|\psi|^2)$', fontsize=12)
    if i == 0:
        ax.set_ylabel(r'(a) Deviation $Z$', fontsize=12)
    ax.set_title(case['name'], fontsize=12)
    ax.set_xticks(case['rsi_ranks'])
    ax.grid(True, alpha=0.3, linestyle='--')
    
    ax.legend(fontsize=9, framealpha=0.9, loc='upper right')
    ax.set_ylim([case['dir_rel_error'][-1]/10,1])
    if i < 3:
        ax.set_yticks([1e-7,1e-5,1e-3,1e-1])
    else:
        ax.set_yticks([1e-6,1e-4,1e-2,1])


# Second row: Runtime comparison bar plot
ax_runtime = fig.add_subplot(gs[1, :3])

# Prepare data for grouped bar chart
x = np.arange(len(test_cases))
width = 0.15

rsi_runtimes = [case['rsi_runtime'] if case['rsi_runtime'] > 0 else np.nan for case in test_cases]
dir_runtimes = [case['dir_runtime'] if case['dir_runtime'] > 0 else np.nan for case in test_cases]
chi_values = [20, 40, 60, 80, 100]

bars1 = ax_runtime.bar(x - width/2, rsi_runtimes, width, label='RSI', 
                       color=color_rsi, alpha=0.8, edgecolor='black', linewidth=1)
bars2 = ax_runtime.bar(x + width/2, dir_runtimes, width, label='Direct', 
                       color=color_dir, alpha=0.8, edgecolor='black', linewidth=1)

# Add line plots connecting the bar data points
#ax_runtime.plot(x - width/2, rsi_runtimes, '--', color=color_rsi, 
#               linewidth=2, markersize=6, alpha=0.7, zorder=3)
#ax_runtime.plot(x + width/2, dir_runtimes, '--', color=color_dir, 
#               linewidth=2, markersize=6, alpha=0.7, zorder=3)

# Add reference lines for chi^3 and chi^4 scaling
chi_range = np.linspace(chi_values[0], chi_values[-1], 100)
x_range = np.linspace(x[0], x[-1], 100)

# Scale reference lines to align with the data
scale_chi3 = rsi_runtimes[0] / (chi_values[0]**3)
scale_chi4 = dir_runtimes[0] / (chi_values[0]**4)

chi3_line = (scale_chi3-0.15) * (chi_range ** 3)
chi4_line = scale_chi4 * (chi_range ** 4)

ax_runtime.plot(x_range, chi3_line, ':', color=color_rsi, linewidth=2.5, 
               label=r'$\propto \chi^3$', alpha=0.7, zorder=2)
ax_runtime.plot(x_range, chi4_line, ':', color=color_dir, linewidth=2.5, 
               label=r'$\propto \chi^4$', alpha=0.7, zorder=2)

ax_runtime.set_xlabel(r'Input $\chi_{\max}(\psi)$', fontsize=13)
ax_runtime.set_ylabel('(b) Runtime (ms)', fontsize=12)
ax_runtime.set_yscale('log')
ax_runtime.set_ylim([1e2, 1e7])
ax_runtime.set_xticks(x)
ax_runtime.set_xticklabels(['20', '40', '60', '80', '100'], fontsize=13)
ax_runtime.legend(fontsize=11, framealpha=0.9, loc='upper left')
ax_runtime.grid(True, alpha=0.3, axis='y', linestyle='--', which='both')

# Add value labels on bars
def autolabel(bars, ax):
    for bar in bars:
        height = bar.get_height()
        if height > 0 and not np.isnan(height):
            # Format large numbers with scientific notation
            if height >= 10000:
                label = f'{height:.2e}'
            else:
                label = f'{height:.0f}'
            ax.annotate(label,
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

autolabel(bars1, ax_runtime)
autolabel(bars2, ax_runtime)

# Add speedup subplot
ax_speedup = fig.add_subplot(gs[1, 3:5])

# Calculate speedup
speedup = [dir_runtimes[i] / rsi_runtimes[i] for i in range(len(test_cases))]

# Plot speedup
ax_speedup.plot(chi_values, speedup, 'o-', color='green', linewidth=2.5, 
               markersize=10, markeredgewidth=1.5, markeredgecolor='white', label='Speedup')

ax_speedup.set_xlabel(r'Input $\chi_{\max}(\psi)$', fontsize=13)
ax_speedup.set_ylabel('(c) Speedup of RSI', fontsize=12)
ax_speedup.set_ylim([1, 90])
ax_speedup.set_xticks(chi_values)
ax_speedup.set_xticklabels(['20', '40', '60', '80', '100'], fontsize=13)
ax_speedup.grid(True, alpha=0.3, linestyle='--')

# Add value labels on points
for i, (chi, speed) in enumerate(zip(chi_values, speedup)):
    ax_speedup.annotate(f'{speed:.1f}×',
                       xy=(chi, speed),
                       xytext=(0, 8),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('convergence_runtime_comparison.pdf', dpi=300, bbox_inches='tight')
plt.show()