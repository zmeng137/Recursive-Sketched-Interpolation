import numpy as np
import matplotlib.pyplot as plt

B = ["B=5", "B=10", "B=15", "B=20"]
target_rank_rmax = [r"($\chi_{\max}(g^{\text{RSI}})=20$)", r"($\chi_{\max}(g^{\text{RSI}})=30$)", r"($\chi_{\max}(g^{\text{RSI}})=40$)", r"($\chi_{\max}(g^{\text{RSI}})=50$)"]

sketch_dim = {}
sketch_dim["B=5"]  = [1, 5, 10, 15, 20]
sketch_dim["B=10"] = [5, 10, 15, 20, 25]
sketch_dim["B=15"] = [5, 10, 15, 20, 25, 30]
sketch_dim["B=20"] = [10, 15, 20, 25, 30, 35]

error = {}
error["B=5"] =  [5.39e-1, 1.90e-3, 1.56e-7, 2.77e-7, 1.75e-7]  
error["B=10"] = [1.96e-1, 1.34e-2, 3.78e-4, 6.22e-4, 2.49e-4]
error["B=15"] = [4.47e-1, 2.37e-2, 3.22e-3, 1.24e-4, 1.01e-4, 1.18e-4]
error["B=20"] = [2.28e-2, 3.61e-3, 7.30e-4, 1.06e-5, 2.31e-5, 1.19e-5]

runtime = {}
runtime["B=5"] =  [7.83, 38.94, 91.26, 101.12, 109.26]
runtime["B=10"] = [26.66, 90.95, 102.56, 110.02, 119.23]
runtime["B=15"] = [29.31, 95.92, 112.26, 128.84, 134.07, 140.34]
runtime["B=20"] = [102.12, 113.79, 135.76, 170.45, 186.36, 190.65]

colors = ["#DFA06E", "#84D394", "#94C4E4", "#DF5F68"]
markers = ['o', 's', '^', 'd']

# Create 2x1 subplot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Error vs Sketch Dimension
for i, b in enumerate(B):
    ax1.semilogy(sketch_dim[b], error[b], marker=markers[i], color=colors[i], 
                 linewidth=2, markersize=8, label=b+target_rank_rmax[i])

ax1.set_xlabel(r'Sketching Dimension $k$', fontsize=18)
ax1.set_ylabel(r'Relative Error $\epsilon_r$', fontsize=18)
ax1.legend(fontsize=15)
ax1.tick_params(axis='x', labelsize=16)
ax1.tick_params(axis='y', labelsize=16)  
ax1.grid(True, alpha=0.3)

# Plot 2: Runtime vs Sketch Dimension
for i, b in enumerate(B):
    ax2.plot(sketch_dim[b], runtime[b], marker=markers[i], color=colors[i], 
             linewidth=2, markersize=8, label=b+target_rank_rmax[i])

ax2.set_xlabel(r'Sketching Dimension $k$', fontsize=18)
ax2.set_ylabel('Runtime (ms)', fontsize=18)
ax2.legend(fontsize=15)
ax2.tick_params(axis='x', labelsize=16)
ax2.tick_params(axis='y', labelsize=16)  
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('error_runtime_vs_sketch_dim.pdf', format='pdf', dpi=300, bbox_inches='tight')
print("Saved: error_runtime_vs_sketch_dim.pdf")

plt.show()


# For B=20, vary target chi_max
B20_err_skdim = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
B20_rmax = [r"$\chi_{\max}(g^{\text{RSI}})=30$",r"$\chi_{\max}(g^{\text{RSI}})=40$",r"$\chi_{\max}(g^{\text{RSI}})=50$",r"$\chi_{\max}(g^{\text{RSI}})=60$"]

B20_err_sk = {}
B20_err_sk[B20_rmax[0]] = [4.08e-2, 2.38e-2, 2.34e-3, 3.87e-3, 7.89e-3, 3.26e-3, 4.99e-3, 1.83e-3, 9.75e-4, 3.41e-3, 2.45e-3]
B20_err_sk[B20_rmax[1]] = [9.51e-2, 4.04e-3, 1.05e-3, 7.41e-4, 7.77e-4, 1.98e-4, 4.20e-4, 3.74e-4, 2.22e-4, 1.56e-3, 3.14e-4]
B20_err_sk[B20_rmax[2]] = [3.09e-2, 1.77e-2, 6.88e-4, 2.39e-5, 1.68e-5, 8.54e-6, 8.41e-6, 1.13e-5, 1.39e-5, 1.23e-5, 1.48e-5]
B20_err_sk[B20_rmax[3]] = [2.02e-2, 5.04e-3, 7.70e-4, 8.84e-6, 5.71e-7, 3.56e-7, 3.87e-7, 4.14e-7, 4.60e-7, 4.79e-7, 5.19e-7]

plt.figure(figsize=(7, 5))
for i, b in enumerate(B20_rmax):
    plt.semilogy(B20_err_skdim, B20_err_sk[b], marker=markers[i], color=colors[i], 
             linewidth=2, markersize=8, label=B20_rmax[i])

plt.xlabel(r'Sketching Dimension $k$', fontsize=18)
plt.ylabel(r'Relative Error $\epsilon_r$', fontsize=18)
plt.ylim([1e-7,1])
#plt.title('Runtime vs Sketch Dimension', fontsize=14)
plt.legend(fontsize=13)
plt.annotate("B=20",horizontalalignment='center', verticalalignment='top',xy=(35, 0.1), fontsize=20)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)  
plt.tight_layout()
plt.savefig('error_vs_sketch_dim_varying_rmax.pdf', format='pdf', dpi=300, bbox_inches='tight')
print("Saved: error_vs_sketch_dim_varying_rmax.pdf")
