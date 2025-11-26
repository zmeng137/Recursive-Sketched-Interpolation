import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec

# Statistics of random tensor train (10,10,10,10,10,10,10) (tensorly randomtt)
#shape = [10, 10, 10, 10, 10, 10, 10]
#ttrank_tt1 = [1, 10, r_max, r_max, r_max, r_max, r_max, 1]
#ttrank_tt2 = [1, 10, r_max, r_max, r_max, r_max, r_max, 1]

# RSI setting
#contract_number = 2; rg_max = 10; seed = 10; eps=0; sketch_dim = 5
#Direct + TT rounding rg_max = 10

# Let's first fix our target rank at rg_max = 10 and see...
rmax_f = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
rsi_time = [63.33, 65.96, 65.90, 92.39, 105.53,
            124.24, 175.04, 136.91, 131.81, 234.09]  # milliseconds
direct_time = [1.97, 23.42, 95.46, 196.42, 413.12,
               816.44, 1452.10, 2379.50, 3752.84, 5640.23]
rounding_time = [160.16, 535.64, 1775.66, 4554.28, 9421.69, 
                 18137.75, 33874.10, 71334.83, 106002.32, 201426.91]

rsi_relerr = [6.12e-3, 2.15e-3, 1.34e-3, 9.32e-4, 7.92e-4,
              8.19e-4, 3.76e-4, 2.10e-3, 3.19e-4, 2.50e-4]
direct_relerr = [3.41e-16, 3.27e-16, 3.36e-16, 3.16e-16, 3.27e-16,
                 3.32e-16, 3.34e-16, 3.11e-16, 3.42e-16, 3.22e-16]  # Note that the direct method without compression is exact but square the input tt rank
rounding_relerr = [1.95e-3, 8.51e-4, 5.16e-4, 3.33e-4, 2.36e-4,
                   1.82e-4, 1.47e-4, 1.22e-4, 1.00e-4, 9.12e-5]

# Target rank data
rsi_target_rank = [20] * len(rmax_f)  # All 20
direct_target_rank = [r**2 for r in rmax_f]  # Square of rmax_f
rsi_recompress_tm = [0] * len(rmax_f)

direct_color = "#659DC2"
rounding_color = "#8CCC78"
rsi_color = "#CC5551"

# Create figure with GridSpec
fig = plt.figure(figsize=(20, 4))
gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.2)

# First subplot: Runtime vs input rank
ax1 = fig.add_subplot(gs[:, 0])
ax1.plot(rmax_f, rsi_time, label='RSI', color=rsi_color, marker='o')
ax1.plot(rmax_f, direct_time, label='Direct', color=direct_color, marker='s')
ax1.set_xlabel(r"Input maximum bond dimension $\chi_{f,\max}$")
ax1.set_ylabel("Runtime (ms)")
ax1.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
ax1.grid()
ax1.legend(loc='upper left')
ax1.set_title("Runtime of Hadamard Product")

# Enable scientific notation
formatter1 = ScalarFormatter(useMathText=True)
formatter1.set_scientific(True)
formatter1.set_powerlimits((-1, 1))
ax1.yaxis.set_major_formatter(formatter1)

# Second subplot (top): Target rank vs input rank
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(rmax_f, rsi_target_rank, label='RSI', 
         color=rsi_color, marker='o')
ax2.plot(rmax_f, direct_target_rank, label='Direct', 
         color=direct_color, marker='s')
ax2.set_ylabel(r"Result $\chi_{g,\max}$")
ax2.grid()
ax2.legend(loc='upper left')
ax2.set_title("Output TT's Rank and Relative Error")
# Enable scientific notation
formatter2 = ScalarFormatter(useMathText=True)
formatter2.set_scientific(True)
formatter2.set_powerlimits((-1, 1))
ax2.yaxis.set_major_formatter(formatter2)

# Second subplot (bottom): Relative error vs input rank
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(rmax_f, rsi_relerr, label='RSI', color=rsi_color, marker='o')
ax3.plot(rmax_f, direct_relerr, label='Direct', color=direct_color, marker='s')
ax3.plot(rmax_f, rounding_relerr, label='Direct + Rounding', color=rounding_color, marker='^')
ax3.set_yscale("log", base=10)
ax3.set_xlabel(r"Input maximum bond dimension $\chi_{f,\max}$")
ax3.set_ylabel("Relative Error")
ax3.grid()
ax3.legend(loc='center right')
ax3.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
ax3.set_yticks([1e-16, 1e-3, 1e0])


# Third subplot: Placeholder
ax4 = fig.add_subplot(gs[:, 2])
ax4.plot(rmax_f, rsi_recompress_tm, label='Recompression after RSI', color=rsi_color, marker='o')
ax4.plot(rmax_f, rounding_time, label='Recompression after Direct (TT rounding)', color=rounding_color, marker='^')
ax4.set_ylabel("Runtime (ms)")
ax4.set_title("Recompression Time after Product")
ax4.grid()
ax4.legend()
ax4.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
ax4.set_xlabel(r"Input maximum bond dimension $\chi_{f,\max}$")
ax4.yaxis.set_major_formatter(formatter2)

fig.savefig("timescaling_randtt.png", dpi=150, bbox_inches='tight')
fig.savefig("timescaling_randtt.pdf", dpi=150, bbox_inches='tight')