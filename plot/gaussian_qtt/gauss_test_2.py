import numpy as np
import matplotlib.pyplot as plt

# Use broader Gaussians that overlap
w1 = 0.01  
x1,x2 = 0.49,0.51
f1 = lambda x: np.exp(-(x-x1)**2/(2*w1**2))
f2 = lambda x: np.exp(-(x-x2)**2/(2*w1**2))
g = lambda x: f1(x) * f2(x)

# Create x values for evaluation
x = np.linspace(0, 1, 1000)

# Evaluate functions
F1 = f1(x)
F2 = f2(x)
G = g(x)

# Error data
rank = [4,6,8,10,12]
rsi_error = [1.31e-3, 1.48e-6, 1.98e-9, 6.49e-13, 2.86e-14]
tci_error = [7.07e-1, 7.07e-1, 7.07e-1, 7.07e-1, 7.07e-1]
dir_error = [2.56e-4, 6.52e-7, 4.83e-10, 1.16e-13, 9.46e-15]

# Create the figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Define colors
f1_color = "#A4C2A8"
f2_color = "#ACEB98"
g_color = "#C6605C"

rsi_color = "#C6605C"
tci_color = "#6ECF8B"
dir_color = "#597ABA"

# First subplot: Function shapes
ax1 = axes[0]
ax1.plot(x, F1, linewidth=2, label=r'$f_1$', color=f1_color)
ax1.plot(x, F2, linewidth=2, label=r'$f_2$', color=f2_color)
ax1.plot(x, G, linewidth=2.5, label=r'$g=f_1f_2$', color=g_color)
ax1.set_xlabel('x', fontsize=17)
ax1.set_ylabel('(a) Function Value', fontsize=16)
ax1.legend(fontsize=15)
ax1.grid(True, alpha=0.3)

# Second subplot: Error plot
ax2 = axes[1]
ax2.plot(rank, rsi_error, 'o-', linewidth=2, markersize=5, color=rsi_color, label="RSI")
#ax2.plot(rank, tci_error, 's-', linewidth=2, markersize=5, color=tci_color, label="TCI")
ax2.plot(rank, dir_error, '^-', linewidth=2, markersize=5, color=dir_color, label="direct")
ax2.set_xlabel(r'$\chi_{\max}(g)$', fontsize=17)
ax2.set_ylabel(r'(b) Relative Error $\epsilon_r$', fontsize=16)
ax2.set_yscale('log')
ax2.legend(fontsize=16)
ax2.grid(True, alpha=0.3)


plt.subplots_adjust(wspace=0.3)
plt.savefig('gaussian_spike.pdf', dpi=300, bbox_inches='tight')
