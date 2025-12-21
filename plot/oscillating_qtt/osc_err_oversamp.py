import numpy as np
import matplotlib.pyplot as plt

colors = ["#DFA06E", "#84D394", "#94C4E4", "#DF5F68"]
markers = ['o', 's', '^', 'd']

# For B=20, vary target chi_max and oversampling (p for k=chi/2+p)
oversampling = [0, 5, 10, 15, 20]

B20_rmax = [r"$\chi_{\max}(g)=5$",
            r"$\chi_{\max}(g)=10$",
            r"$\chi_{\max}(g)=20$",
            r"$\chi_{\max}(g)=30$"]

B20_err_sk = {}
B20_err_sk[B20_rmax[0]] = [5.72e-1, 5.55e-2, 5.12e-2, 5.22e-2, 4.21e-2]
B20_err_sk[B20_rmax[1]] = [8.84e-3, 3.12e-3, 3.04e-3, 4.01e-3, 3.01e-3]
B20_err_sk[B20_rmax[2]] = [1.25e-5, 6.97e-6, 1.62e-6, 3.62e-6, 2.96e-6]
B20_err_sk[B20_rmax[3]] = [6.19e-9, 5.12e-9, 6.18e-9, 5.99e-9, 5.41e-9]

plt.figure(figsize=(10, 6))
for i, b in enumerate(B20_rmax):
    plt.semilogy(oversampling, B20_err_sk[b], marker=markers[i], color=colors[i], 
             linewidth=2, markersize=8, label=B20_rmax[i])

plt.xlabel(r'Sketch Oversampling $p$', fontsize=18)
plt.ylabel(r'Relative Error $\epsilon_r$', fontsize=18)
plt.ylim([1e-9,1])
#plt.title('Runtime vs Sketch Dimension', fontsize=14)
plt.legend(fontsize=14, loc='upper right',bbox_to_anchor=(1.0, 0.7))
#plt.annotate(r"B=20 for $f_1,f_2$",horizontalalignment='center', verticalalignment='top',xy=(10, 0.1), fontsize=21)
plt.grid(True, alpha=0.3)
plt.xticks(oversampling,fontsize=16)
plt.yticks(fontsize=16)  
plt.tight_layout()
plt.savefig('osc_error_vs_oversamp.pdf', format='pdf', dpi=300, bbox_inches='tight')
print("Saved: osc_error_vs_oversamp.pdf")
