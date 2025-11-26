import numpy as np
import matplotlib.pyplot as plt

B = [5, 10, 15, 20]

target_maxrank = {}
target_maxrank['B=5'] = [5,10,15,20]
target_maxrank['B=10'] = [10,15,20,25,30] 
target_maxrank['B=15'] = [15,20,25,30,35,40]
target_maxrank['B=20'] = [20,25,30,35,40,45,50]

err_rsi = {}
time_rsi = {}

err_rsi['B=5'] = [4.35e-2, 2.16e-3, 4.64e-5, 3.99e-7]
err_rsi['B=10'] = [1.99e-1, 8.83e-2, 4.26e-3, 1.04e-3, 2.98e-4]
err_rsi['B=15'] = [2.99e-1, 1.74e-2, 1.93e-3, 7.19e-4, 2.51e-4, 9.44e-5]
err_rsi['B=20'] = [6.55e-2, 4.34e-3, 3.41e-3, 4.17e-4, 1.62e-4, 8.88e-5, 6.14e-6] 

# Sketch_dim = r_max
time_rsi['B=5'] = [18.16, 28.47, 37.22, 43.53]
time_rsi['B=10'] = [36.98, 52.56, 66.33, 83.55, 98.62]
time_rsi['B=15'] = [60.80, 88.17, 95.76, 114.39, 134.20, ]
time_rsi['B=20'] = [89.46, 108.78, 130.34, 188.80, 174.45, 198.47, 250.22]

err_direct = {}
time_direct = {}

err_direct['B=5'] = [2.16e-3, 3.01e-9, 5.36e-11, 5.36e-11]
err_direct['B=10'] = [2.38e-2, 7.14e-3, 1.64e-4, 3.09e-5, 7.35e-7]
err_direct['B=15'] = [3.33e-3, 2.18e-4, 4.68e-5, 7.68e-6, 1.76e-7, 9.51e-8]
err_direct['B=20'] = [6.10e-4, 8.27e-5, 3.02e-5, 4.24e-7, 3.24e-7, 7.11e-8, 2.86e-9] 

time_direct['B=5'] = [36.85, 38.98, 39.54, 43.88]
time_direct['B=10'] = [414.95, 413.01, 427.02, 451.68, 462.37]
time_direct['B=15'] = [1621.63, 1586.88, 1659.44, 1673.82, 1688.90, 1724.04]
time_direct['B=20'] = [5573.62, 5421.36, 5844.70, 6311.23, 5741.52, 5647.26, 6432.21]

colors = ["#DFA06E", "#84D394", "#94C4E4", "#DF5F68"]
markers = ['o', 's', '^', 'd']

# Figure 1: Error comparison
plt.figure(figsize=(7.5, 4.5))


for i, b in enumerate(B):
    b_str = f'B={b}'
    h1, = plt.semilogy(target_maxrank[b_str], err_rsi[b_str], marker=markers[i], 
                 color=colors[i], linewidth=2, markersize=8, 
                 linestyle='-', label = f'{b_str} (RSI)')

for i, b in enumerate(B):
    b_str = f'B={b}'
    h2, = plt.semilogy(target_maxrank[b_str], err_direct[b_str], marker=markers[i], 
                color=colors[i], linewidth=2, markersize=8, 
                linestyle='--', label = f'{b_str} (Direct)')
  
plt.xlabel(r'$\chi_{\max}(g^{\text{RSI}})$', fontsize=18)
plt.ylabel(r'Relative Error $\epsilon_r$', fontsize=18)
plt.legend(
#    bbox_to_anchor=(1,0.1),
    loc='lower right',          # Anchor the legend's left side to a specific point
    ncol=2,                     # Display legend entries in 2 columns
    fontsize=13,
    framealpha=0.3
)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)  
plt.tight_layout()
plt.savefig('error_comparison_rsi_direct.pdf', format='pdf', dpi=300, bbox_inches='tight')
print("Saved: error_comparison_rsi_direct.pdf")

# Figure 2: Runtime comparison (bar chart with lines) - LOG SCALE
plt.figure(figsize=(7.5, 4.5))

# Get last runtime value for each B

last_time_direct = [time_direct[f'B={b}'][-1] for b in B]

last_time_rsi = [65.17, 94.77, 120.27, 167.91] # (r_max,sketch_dim) = (20,20), (30,25), (40,30), (50,35)

x = np.arange(len(B))
width = 0.35

# Create bars
bars1 = plt.bar(x - width/2, last_time_direct, width, label='Direct', 
                color='#84D394', alpha=0.8, edgecolor='black', linewidth=1.2)
bars2 = plt.bar(x + width/2, last_time_rsi, width, label='RSI', 
                color='#DF5F68', alpha=0.8, edgecolor='black', linewidth=1.2)

# Add lines connecting the bars
plt.plot(x - width/2, last_time_direct, marker='o', color='#2d5f3d', 
         linewidth=2, markersize=8, linestyle='--', label='_nolegend_')
plt.plot(x + width/2, last_time_rsi, marker='s', color='#8b2428', 
         linewidth=2, markersize=8, linestyle='--', label='_nolegend_')

target_chi_max = [20, 30, 40, 50]
for i in range(len(B)):
    max_height = max(last_time_rsi[i], last_time_direct[i])
    plt.text(x[i]-0.15, max_height * 1.5, 
             f'B={B[i]}\n' + r'$\chi_{\max}(g^{\text{RSI}})$=' + f'{target_chi_max[i]}',
             ha='center', va='bottom', fontsize=12)


plt.xlabel(r'Input $\chi_{\max}(f_1)$ and $\chi_{\max}(f_2)$', fontsize=18)
plt.ylabel('Runtime (ms)', fontsize=18)
plt.yticks(fontsize=18)  
plt.yscale('log')  # Add log scale to y-axis
plt.ylim(10, 30000)

input_rank = [10,20,30,40]
plt.xticks(x, [f"{b}" for b in input_rank], fontsize=18)
plt.legend(fontsize=17,loc='upper left')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('runtime_comparison_rsi_direct.pdf', format='pdf', dpi=300, bbox_inches='tight')
print("Saved: runtime_comparison_rsi_direct.pdf")

plt.show()