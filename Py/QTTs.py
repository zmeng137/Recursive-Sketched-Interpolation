import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt

from tt_svd import TT_SVD
from tensor_cross import TT_CUR_L2R, cross_core_interp_assemble, TCI_2site, cross_inv_merge
from rank_revealing import prrldu

# Let's say we have a 5 order quantics
# numpy.fromfunction
def populate_tensor_fromfunction(dims, func):
    # Populate tensor using numpy.fromfunction
    def array_func(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
        return func(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)    
    # Use fromfunction to create the tensor
    tensor_data = np.fromfunction(array_func, dims, dtype=int)
    return tl.tensor(tensor_data)

def scatter_plot_f1f2(x_tensor, f1_tensor, f2_tensor, g_tensor):
    scatter = plt.scatter(x_tensor, f1_tensor, 
                        s=8,          # Point size
                        alpha=0.8,  
                        edgecolors='black',  # Point borders
                        linewidth=0.5,
                        label='f1')
    scatter = plt.scatter(x_tensor, f2_tensor, 
                        s=8,          # Point size
                        alpha=0.8,  
                        edgecolors='black',  # Point borders
                        linewidth=0.5,
                        label='f2')
    scatter = plt.scatter(x_tensor, g_tensor, 
                        s=15,          # Point size
                        alpha=0.8,  
                        edgecolors='black',  # Point borders
                        linewidth=0.5,
                        label='g')
    plt.legend()
    plt.savefig("quantics.png")
    return

quantic_repres = lambda x1,x2,x3,x4,x5,x6,x7,x8,x9,x10: x1/2 + x2/(2**2) + x3/(2**3) + x4/(2**4) + x5/(2**5) + x6/(2**6) + x7/(2**7) + x8/(2**8) + x9/(2**9) + x10/(2**10)
func1 = lambda t: -0.3 *t ** 6 + 2 * t ** 2 + 0.6 * t - 1 + 0.4 * np.sin(-4 * np.pi * t)
#t ** 5 - 3 * t ** 3 + 10 * t -6 #5 * np.sin(-2 * np.pi * t) - 3 * np.exp(t)
func2 = lambda t: 0.6 * t ** 4 - 0.1 * t ** 7 + 5 -1.2 * np.cos(6 * np.pi * t) - 2 * t ** 3 + 4 
#-10 * np.exp(-(t - 1) * (t - 1) / 2) - 2 * t ** 3 + 4 
g_func = lambda t: func1(t) * func2(t)

shape = (2,2,2,2,2,2,2,2,2,2)
dim = len(shape)
x_tensor = populate_tensor_fromfunction(shape, quantic_repres)
f1_tensor = func1(x_tensor)
f2_tensor = func2(x_tensor)
g_tensor = f1_tensor * f2_tensor
#scatter_plot_f1f2(x_tensor, f1_tensor, f2_tensor, g_tensor)

''' === Create initial (rank-1) interpolation I/J sets === '''
r_max = 1
eps = 1e-14
TTCores, TTCores_Cross, TTRank, Nested_I_rank1, Nested_J_rank1 = TT_CUR_L2R(f1_tensor, r_max, eps)
Assemble_TTCore_Cross = cross_core_interp_assemble(f1_tensor, Nested_I_rank1, Nested_J_rank1, TTRank)
for i in range(2 * len(TTRank) - 3):
    diff_flag = (Assemble_TTCore_Cross[i] == TTCores_Cross[i]).all()
    if (diff_flag == False):
        print(f"Interpolation assembly error at {i}!")
Nested_I_rank1[0] = []
Nested_J_rank1[dim+1] = []

''' === Tensor cross interpolation for f1, f2, g === '''
# TCI-2site of f1
r_max = 3
interp_I_f1, interp_J_f1, TTRank_f1, recon_f1 = TCI_2site(f1_tensor, 1e-10, r_max, Nested_I_rank1, Nested_J_rank1)
error = tl.norm(f1_tensor - recon_f1) / tl.norm(f1_tensor)
print(f"Relative error of f1 QTT at r_max = {r_max}: {error}")

# TCI-2site of f2
r_max = 3
interp_I_f2, interp_J_f2, TTRank_f2, recon_f2 = TCI_2site(f2_tensor, 1e-10, r_max, Nested_I_rank1, Nested_J_rank1)
error = tl.norm(f2_tensor - recon_f2) / tl.norm(f2_tensor)
print(f"Relative error of f2 QTT at r_max = {r_max}: {error}")

# TCI-2site of g
r_max = 4
interp_I_g, interp_J_g, TTRank_g, recon_g = TCI_2site(g_tensor, 1e-10, r_max, Nested_I_rank1, Nested_J_rank1)
error = tl.norm(g_tensor - recon_g) / tl.norm(g_tensor)
print(f"Relative error of g QTT at r_max = {r_max}: {error}")

# Assmeble g using f1's and f2's interpolation individually
TT_cross = cross_core_interp_assemble(g_tensor, interp_I_f1, interp_J_f1, TTRank_f1)
TT_cores = cross_inv_merge(TT_cross, dim)
recon_g_Interpf1 = tl.tt_to_tensor(TT_cores)
error = tl.norm(g_tensor - recon_g_Interpf1) / tl.norm(g_tensor)
print(f"Relative error of g QTT (using f1 Interp) at r_max = {r_max}: {error}")

TT_cross = cross_core_interp_assemble(g_tensor, interp_I_f2, interp_J_f2, TTRank_f2)
TT_cores = cross_inv_merge(TT_cross, dim)
recon_g_Interpf2 = tl.tt_to_tensor(TT_cores)
error = tl.norm(g_tensor - recon_g_Interpf2) / tl.norm(g_tensor)
print(f"Relative error of g QTT (using f2 Interp) at r_max = {r_max}: {error}")

# For the first 

''' === Plot TCI of g (assembled from f1, f2, g interp) === '''
# Plot of exact g 
x_axis = np.linspace(0,1,100)
exact_g = g_func(x_axis)
plt.plot(x_axis, exact_g, 'r-', linewidth=1, label='Exact g(x)')

# Plot of reconstruction of TCI of g
scatter = plt.scatter(x_tensor, recon_g, 
                        s=1,          # Point size
                        alpha=0.8,  
                        linewidth=0.25,
                        label='recon g (TCI g)')
# Plot of reconstructed g from f1 interpolation
scatter = plt.scatter(x_tensor, recon_g_Interpf1, 
                        s=1,          # Point size
                        alpha=0.8,  
                        linewidth=0.25,
                        label='recon g (f1 interp)')
# Plot of reconstructed g from f2 interpolation
scatter = plt.scatter(x_tensor, recon_g_Interpf2, 
                        s=1,          # Point size
                        alpha=0.8,  
                        linewidth=0.25,
                        label='recon g (f2 interp)')
plt.legend()
plt.savefig("quantics.png")
pass

# TCI of QTT g(x)
# Create initial I,J sets
r_max = 1
eps = 1e-14
TTCores, TTCores_Cross_3, TTRank, Nested_I_3, Nested_J_3 = TT_CUR_L2R(g_tensor, r_max, eps)
recon = tl.tt_to_tensor(TTCores)
error = tl.norm(g_tensor - recon) / tl.norm(g_tensor)
print(f"Relative error of g=f1*f2 QTT at r_max = {r_max}: {error}")
Assemble_TTCore_Cross = cross_core_interp_assemble(g_tensor, Nested_I_3, Nested_J_3, TTRank)
for i in range(2 * len(TTRank) - 3):
    diff_flag = (Assemble_TTCore_Cross[i] == TTCores_Cross_3[i]).all()
    if (diff_flag == False):
        print(f"Interpolation assembly error at {i}!")
Nested_I_3[0] = []
Nested_J_3[11] = []
r_max1= 1
r_max2= 2
r_max3= 3
interp_I, interp_J, TTRank, g_recon1 = TCI_2site(g_tensor, 1e-10, r_max1, Nested_I_3, Nested_J_3)
interp_I, interp_J, TTRank, g_recon2 = TCI_2site(g_tensor, 1e-10, r_max2, Nested_I_3, Nested_J_3)
interp_I, interp_J, TTRank, g_recon3 = TCI_2site(g_tensor, 1e-10, r_max3, Nested_I_3, Nested_J_3)



x_axis = np.linspace(0,1,100)
exact_g = g_func(x_axis)
plt.plot(x_axis, exact_g, 'r-', linewidth=1, label='Exact g(x)')

scatter = plt.scatter(x_tensor, g_recon1, 
                        s=1,          # Point size
                        alpha=0.8,  
                        linewidth=0.25,
                        label=f'rank={r_max1}')
scatter = plt.scatter(x_tensor, g_recon2, 
                        s=1,          # Point size
                        alpha=0.8,  
                        linewidth=0.25,
                        label=f'rank={r_max2}')
scatter = plt.scatter(x_tensor, g_recon3, 
                        s=1,          # Point size
                        alpha=0.8,  
                        linewidth=0.25,
                        label=f'rank={r_max3}')
plt.legend()
plt.savefig("quantics.png")




