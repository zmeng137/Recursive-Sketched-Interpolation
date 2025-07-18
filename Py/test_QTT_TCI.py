import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt

from QTT import populate_tensor_fromfunction, union_rows_bounded, scatter_plot_f1f2, integral_qtt, value_query_QTT
from tensor_cross import TT_CUR_L2R, cross_core_interp_assemble, TCI_2site, cross_inv_merge, TCI_union_two

''' === Quantics representation construction === '''
# Quantics construction
quantic_repres = lambda x1,x2,x3,x4,x5,x6,x7,x8,x9,x10: x1/2 + x2/(2**2) + x3/(2**3) + x4/(2**4) + x5/(2**5) + x6/(2**6) + x7/(2**7) + x8/(2**8) + x9/(2**9) + x10/(2**10)
func1 = lambda t: 1.2 * t ** 6 - 1.2 * np.sqrt(t) - 1 + 0.6 * np.sin(10.3 * np.pi * t)  #t ** 5 - 3 * t ** 3 + 10 * t -6 #5 * np.sin(-2 * np.pi * t) - 3 * np.exp(t)
func2 = lambda t: -1.1 * t ** 7 - 12 + np.exp(3.1*t) - 0.81 * np.cos(6 * np.pi * t) - 2 * t ** 2 + 4 + np.tan(t)  #-10 * np.exp(-(t - 1) * (t - 1) / 2) - 2 * t ** 3 + 4 
#func1 = lambda t: np.exp((t-0.2)*(t-0.2)/0.001)
g_func = lambda t: func1(t) * func2(t)
shape = (2,2,2,2,2,2,2,2,2,2)
dim = len(shape)
x_tensor = populate_tensor_fromfunction(shape, quantic_repres)
f1_tensor = func1(x_tensor)
f2_tensor = func2(x_tensor)
g_tensor = f1_tensor * f2_tensor
scatter_plot_f1f2(x_tensor, g_tensor, f1_tensor, f2_tensor)
pass


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
r_max = 7
interp_I_f1, interp_J_f1, TTRank_f1, recon_f1 = TCI_2site(f1_tensor, 1e-10, r_max, Nested_I_rank1, Nested_J_rank1)
error = tl.norm(f1_tensor - recon_f1) / tl.norm(f1_tensor)
print(f"Relative error of f1 QTT at r_max = {r_max}: {error}")

# TCI-2site of f2
r_max = 5
interp_I_f2, interp_J_f2, TTRank_f2, recon_f2 = TCI_2site(f2_tensor, 1e-10, r_max, Nested_I_rank1, Nested_J_rank1)
error = tl.norm(f2_tensor - recon_f2) / tl.norm(f2_tensor)
print(f"Relative error of f2 QTT at r_max = {r_max}: {error}")

# TCI-2site of g
r_max = 9
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
print(f"Relative error of g QTT (using f2 Interp) at r_max = {r_max}: {error}\n")

pass


''' === Test of union method === '''
# PRRLU-based UNION method
interp_I_union_g, interp_J_union_g, TTRank_union = TCI_union_two(f1_tensor, interp_I_f1, interp_J_f1, f2_tensor, interp_I_f2, interp_J_f2, 0)
TT_cross = cross_core_interp_assemble(g_tensor, interp_I_union_g, interp_J_union_g, TTRank_union)
TT_cores = cross_inv_merge(TT_cross, dim)
recon_g_prrluUnion = tl.tt_to_tensor(TT_cores)
error_prrluUnion = tl.norm(g_tensor - recon_g_prrluUnion) / tl.norm(g_tensor)
print(f"Recon errror (PRRLU-based UNION Method) =: {error_prrluUnion}")

# Random UNION method
attempt = np.arange(1,101)
rel_error_runion = []
for i in range(len(attempt)):
    interp_I_union_g, interp_J_union_g, TTRank_union = TCI_union_two(f1_tensor, interp_I_f1, interp_J_f1, f2_tensor, interp_I_f2, interp_J_f2, 1)
    TT_cross = cross_core_interp_assemble(g_tensor, interp_I_union_g, interp_J_union_g, TTRank_union)
    TT_cores = cross_inv_merge(TT_cross, dim)
    recon_g_prrluUnion = tl.tt_to_tensor(TT_cores)
    error_randUnion = tl.norm(g_tensor - recon_g_prrluUnion) / tl.norm(g_tensor)
    rel_error_runion.append(error_randUnion)
rel_error_prrluUnion = np.ones(len(attempt)) * error_prrluUnion

r_max = np.max(TTRank_union)-2
interp_I_g, interp_J_g, TTRank_g, recon_g = TCI_2site(g_tensor, 0, r_max, Nested_I_rank1, Nested_J_rank1)
error_prrluTCI = tl.norm(g_tensor - recon_g) / tl.norm(g_tensor)
rel_error_TCI = np.ones(len(attempt)) * error_prrluTCI

plt.figure()
plt.semilogy(attempt, rel_error_runion, label='Random Union', linestyle=':')
plt.semilogy(attempt, rel_error_prrluUnion, label='PRRLU Union')
plt.semilogy(attempt, rel_error_TCI, label='PRRLU TCI', linestyle='--')
plt.grid()
plt.legend()
plt.xlabel("Number of attempts")
plt.ylabel("Relative error")
plt.savefig("rel_error_union.png")

pass


''' === Test the idea of hierarchical integral === '''
# test the integral
TT_cross = cross_core_interp_assemble(g_tensor, interp_I_g, interp_J_g, TTRank_g)
TT_cores = cross_inv_merge(TT_cross, dim)
#pos = [1,0,0,0,0,0,1,1,1,1]
#val = value_query_QTT(TT_cores, TTRank_g, pos)
integral_qtci_1 = integral_qtt(TT_cores, dim, 0)
integral_qtci_2 = integral_qtt(TT_cores, dim, 1)
integral_qten = g_tensor.sum() / (2**dim)
error_int_1 = np.abs(integral_qten - integral_qtci_1) / np.abs(integral_qten)
error_int_2 = np.abs(integral_qten - integral_qtci_2) / np.abs(integral_qten)
print(f"Relative error of g QTT integral (vs QTensor integral) {error_int_1}, {error_int_2}")

# Try on hierarchical integral method
m = 5

TT_cross_f1 = cross_core_interp_assemble(f1_tensor, interp_I_f1, interp_J_f1, TTRank_f1)
TT_cores_f1 = cross_inv_merge(TT_cross_f1, dim)
integral_qtci_f1 = integral_qtt(TT_cores_f1, m, 1)
lead_TT_int_f1 = TT_cores_f1[0:(dim-m-1)].copy()
last_core = TT_cores_f1[dim-m-1] @ integral_qtci_f1.reshape(-1,1)
lead_TT_int_f1.append(last_core)
TTint_contract_f1 = tl.tt_to_tensor(lead_TT_int_f1)

TT_cross_f2 = cross_core_interp_assemble(f2_tensor, interp_I_f2, interp_J_f2, TTRank_f2)
TT_cores_f2 = cross_inv_merge(TT_cross_f2, dim)
integral_qtci_f2 = integral_qtt(TT_cores_f2, m, 1)
lead_TT_int_f2 = TT_cores_f2[0:(dim-m-1)].copy()
last_core = TT_cores_f2[dim-m-1] @ integral_qtci_f2.reshape(-1,1)
lead_TT_int_f2.append(last_core)
TTint_contract_f2 = tl.tt_to_tensor(lead_TT_int_f2)

TTint_contract_f1f2 = TTint_contract_f1 * TTint_contract_f2

TT_cross_g = cross_core_interp_assemble(g_tensor, interp_I_g, interp_J_g, TTRank_g)
TT_cores_g = cross_inv_merge(TT_cross_g, dim)
integral_qtci_g = integral_qtt(TT_cores_g, m, 1)
lead_TT_int_g = TT_cores_g[0:(dim-m-1)].copy()
last_core = TT_cores_g[dim-m-1] @ integral_qtci_g.reshape(-1,1)
lead_TT_int_g.append(last_core)
TTint_contract_g = tl.tt_to_tensor(lead_TT_int_g)

# Produce nested I/J set for shorter dimension
r_max = 1
eps = 1e-14
TTCores, TTCores_Cross, TTRank, Nested_I_rank1, Nested_J_rank1 = TT_CUR_L2R(TTint_contract_f1f2, r_max, eps)
Assemble_TTCore_Cross = cross_core_interp_assemble(TTint_contract_f1f2, Nested_I_rank1, Nested_J_rank1, TTRank)
for i in range(2 * len(TTRank) - 3):
    diff_flag = (Assemble_TTCore_Cross[i] == TTCores_Cross[i]).all()
    if (diff_flag == False):
        print(f"Interpolation assembly error at {i}!")
Nested_I_rank1[0] = []
Nested_J_rank1[dim-m+1] = []
r_max = 4
interp_I_f1f2_TTint, interp_J_f1f2_TTint, TTRank_f1f2_TTint, recon_f1f2_TTint = TCI_2site(TTint_contract_f1f2, 1e-10, r_max, Nested_I_rank1, Nested_J_rank1)
error = tl.norm(TTint_contract_f1f2 - recon_f1f2_TTint) / tl.norm(TTint_contract_f1f2)
print(f"Relative error of g integral QTT at r_max = {r_max}: {error}")


''' === Need to write function to query g(x) by access QTT of f1 and f2 (chi square really?) === '''



''' === Plot TCI of g (assembled from f1, f2, g interp) === '''
plt.figure()
x_axis = np.linspace(0,1,100)
exact_g = g_func(x_axis)
plt.plot(x_axis, exact_g, 'r-', linewidth=1, label='Exact g(x)')  # Plot of exact g  
plt.scatter(x_tensor, recon_g, s=1, alpha=0.8, linewidth=0.25, label='recon g (TCI g)')  # Plot of reconstruction of TCI of g
plt.scatter(x_tensor, recon_g_Interpf1, s=1, alpha=0.8, linewidth=0.25, label='recon g (f1 interp)')  # Plot of reconstructed g from f1 interpolation
plt.scatter(x_tensor, recon_g_Interpf2, s=1, alpha=0.8, linewidth=0.25, label='recon g (f2 interp)')  # Plot of reconstructed g from f2 interpolation
plt.legend()
plt.grid()
plt.savefig("quantics.png")

pass

# Let's try to refine by selecting optimal area of f1 and f2 in interpolation
# .... I think we still need to do UNION SET to check 
# First to do: Check if I replace some pivots in f1 with f2's pivots (such as x1=0 x2=0 pivots) and then improve quality
#refine_interp_I_f1 = interp_I_f1.copy()
#refine_interp_J_f1 = interp_J_f1.copy()
#refine_TTRank_f1 = TTRank_f1.copy()
for m in range(3,7):
    # Copy lists
    refine_interp_I_f1 = interp_I_f1.copy()
    refine_interp_J_f1 = interp_J_f1.copy()
    refine_TTRank_f1 = TTRank_f1.copy()

    # Union sets
    I_f1 = interp_I_f1[m]
    I_f2 = interp_I_f2[m]
    J_f1 = interp_J_f1[m+1]
    J_f2 = interp_J_f2[m+1]
    I_f1 = union_rows_bounded(I_f1, I_f2, 6)
    J_f1 = union_rows_bounded(J_f1, J_f2, 6)
    
    # Refine interpolation
    refine_interp_I_f1[m] = I_f1
    refine_interp_J_f1[m+1] = J_f1
    refine_TTRank_f1[m] = len(I_f1)

    # Reconstruction
    TT_cross = cross_core_interp_assemble(g_tensor, refine_interp_I_f1, refine_interp_J_f1, refine_TTRank_f1)
    TT_cores = cross_inv_merge(TT_cross, dim)
    recon_g_Interpf1 = tl.tt_to_tensor(TT_cores)
    error = tl.norm(g_tensor - recon_g_Interpf1) / tl.norm(g_tensor)
    print(f"Union I/J at rank {m}(TTRank bounded at 6) =: {error}")
    
# Need to check how J set in x1, x2 dim influence precision? Should be large





TT_cross = cross_core_interp_assemble(g_tensor, interp_I_f1, interp_J_f1, TTRank_f1)
TT_cores = cross_inv_merge(TT_cross, dim)
recon_g_Interpf1 = tl.tt_to_tensor(TT_cores)
error = tl.norm(g_tensor - recon_g_Interpf1) / tl.norm(g_tensor)
print(f"REFINE at r_max = {r_max}: {error}")

plt.figure()
plt.plot(x_axis, exact_g, 'r-', linewidth=1, label='Exact g(x)')
scatter = plt.scatter(x_tensor, recon_g_Interpf1, 
                        s=1,          # Point size
                        alpha=0.8,  
                        linewidth=0.25,
                        label='recon g (f1 REFINE)')
plt.legend()
plt.grid()
plt.savefig("quantics1.png")


# TCI of QTT g(x)
# Create initial I,J sets
r_max = 4
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




