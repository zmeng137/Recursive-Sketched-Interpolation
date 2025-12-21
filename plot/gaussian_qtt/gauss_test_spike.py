import numpy as np
import matplotlib.pyplot as plt

# Use broader Gaussians that overlap
w = 0.1  # Much larger width - less spiky
w1, w2 = w, w
x1 = 0.2
x2 = 0.8  # Closer together so they overlap

f1 = lambda x: np.exp(-(x-x1)**2/(2*w1**2))
f2 = lambda x: np.exp(-(x-x2)**2/(2*w2**2))
g = lambda x: f1(x) * f2(x)

# Generate x values
x = np.linspace(0, 1, 1000)

# Calculate function values
y_f1 = f1(x)
y_f2 = f2(x)
y_g = g(x)

# Create the plot
fig, ax1 = plt.subplots(1, 1, figsize=(5, 3))

# Top plot: all three functions
ax1.plot(x, y_f1, label='f1(x)', linewidth=1, alpha=0.7)
ax1.plot(x, y_f2, label='f2(x)', linewidth=1, alpha=0.7)
ax1.plot(x, y_g, label='g(x) = f1(x) × f1(x)', linewidth=1, color='red')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend()
ax1.grid(True, alpha=0.3)


plt.tight_layout()
plt.savefig("spike_gaussian.png")


eps_tol = [1e-6, 1e-8, 1e-10, 1e-12, 0]
x = [1, 2, 3, 4, 5]
rsi_err = [1.03E-06, 4.43E-09, 1.31E-09, 8.67E-12, 8.30E-13]
tci_err = [7.07E-01, 7.07E-01, 7.07E-01, 7.07E-01, 2.37E-13]

fig, ax1 = plt.subplots(1, 1, figsize=(5, 3))

# Top plot: all three functions
ax1.plot(x, rsi_err, label='rsi', linewidth=1)
ax1.plot(x, tci_err, label='tci', linewidth=1)
ax1.set_xticks(x)
ax1.set_xticklabels(eps_tol)
ax1.set_xlabel('Local ID error tolerance')
ax1.set_ylabel('Reletive error of g=f*f')

ax1.semilogy()
ax1.legend()
ax1.grid(True, alpha=0.3)


plt.tight_layout()
plt.savefig("error_vs_localtol.png")

