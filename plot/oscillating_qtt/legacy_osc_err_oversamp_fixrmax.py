import numpy as np
import matplotlib.pyplot as plt

B = ["B=5", "B=10", "B=15", "B=20"]
target_rank_rmax = [r"($\chi_{\max}(g^{\text{RSI}})=20$)", r"($\chi_{\max}(g^{\text{RSI}})=30$)", r"($\chi_{\max}(g^{\text{RSI}})=40$)", r"($\chi_{\max}(g^{\text{RSI}})=50$)"]

oversampling = [0, 5, 10, 15]

sketch_dim = {}
sketch_dim["B=5"]  = [10, 15, 20, 25]
sketch_dim["B=10"] = [15, 20, 25, 30]
sketch_dim["B=15"] = [20, 25, 30, 35]
sketch_dim["B=20"] = [25, 30, 35, 40]

error = {}
error["B=5"] =  [1.56e-7, 2.77e-7, 1.75e-7, 1.16e-7]  
error["B=10"] = [3.78e-4, 6.22e-4, 2.49e-4, 2.96e-4]
error["B=15"] = [1.24e-4, 1.01e-4, 1.18e-4, 9.16e-5]
error["B=20"] = [1.06e-5, 2.31e-5, 1.19e-5, 1.05e-5]

runtime = {}
runtime["B=5"] =  [7.83, 38.94, 91.26, 101.12, 109.26]
runtime["B=10"] = [26.66, 90.95, 102.56, 110.02, 119.23]
runtime["B=15"] = [29.31, 95.92, 112.26, 128.84, 134.07, 140.34]
runtime["B=20"] = [102.12, 113.79, 135.76, 170.45, 186.36, 190.65]

colors = ["#DFA06E", "#84D394", "#94C4E4", "#DF5F68"]
markers = ['o', 's', '^', 'd']

# Create 2x1 subplot
plt.figure(figsize=(7, 5))

# Plot 1: Error vs Sketch Dimension
for i, b in enumerate(B):
    plt.semilogy(oversampling, error[b], marker=markers[i], color=colors[i], 
                 linewidth=2, markersize=8, label=b+target_rank_rmax[i])

plt.xlabel(r'Sketch Oversampling $p$', fontsize=18)
plt.ylabel(r'Relative Error $\epsilon_r$', fontsize=18)
plt.legend(fontsize=14, loc='upper right')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)  
plt.grid(True, alpha=0.3)
plt.ylim([1e-8,1e-1])

plt.tight_layout()
plt.savefig('osc_error_vs_oversamp_fixrmax.pdf', format='pdf', dpi=300, bbox_inches='tight')
print("Saved: osc_error_vs_oversamp_fixrmax.pdf")

plt.show()