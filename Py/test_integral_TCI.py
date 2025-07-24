import numpy as np
import tensorly as tl
import math as ma
import matplotlib.pyplot as plt

from rank_revealing import prrldu
from interpolation import cur_prrldu
from QTT import populate_tensor_fromfunction, union_rows_bounded, scatter_plot_f1f2, integral_qtt, value_query_QTT
from tensor_cross import TT_CUR_L2R, cross_core_interp_assemble, TCI_2site, cross_inv_merge, TCI_union_two, single_core_interp_assemble

''' === Quantics representation construction === '''
# Quantics construction
quantic_repres = lambda x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12: x1/2 + x2/(2**2) + x3/(2**3) + x4/(2**4) + x5/(2**5) + x6/(2**6) + x7/(2**7) + x8/(2**8) + x9/(2**9) + x10/(2**10) + x11/(2**11) + x12/(2**12)
func1 = lambda t: 1.2 * t ** 4 - 0.2 * np.sqrt(t) - 1 + 0.6 * np.sin(7.3 * np.pi * t)  #t ** 5 - 3 * t ** 3 + 10 * t -6 #5 * np.sin(-2 * np.pi * t) - 3 * np.exp(t)
func2 = lambda t: -1.1 * t ** 7 - 12 + np.exp(3.1*t) - 0.81 * np.cos(6 * np.pi * t) - 2 * t ** 2 + 4 + np.tan(t)  #-10 * np.exp(-(t - 1) * (t - 1) / 2) - 2 * t ** 3 + 4 g_func = lambda t: func1(t) * func2(t)
#func1 = lambda t: 1024 * t**8 * (1-t)**8 * (2*t - 1)**2
#func2 = lambda t: 4096 * t**10 * (1-t)**10 * (1 - 6*t + 6*t**2)**2
shape = (2,2,2,2,2,2,2,2,2,2,2,2)
dim = len(shape)
x_tensor = populate_tensor_fromfunction(shape, quantic_repres)
f1_tensor = func1(x_tensor)
f2_tensor = func2(x_tensor)
g_tensor = f1_tensor * f2_tensor
scatter_plot_f1f2(x_tensor, g_tensor, f1_tensor, f2_tensor)
pass


''' === Tensor cross interpolation for f1, f2, g === '''
# Create initial (rank-1) interpolation I/J sets
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

# TCI-2site of f1
r_max = 5
interp_I_f1, interp_J_f1, TTRank_f1, recon_f1 = TCI_2site(f1_tensor, 0, r_max, Nested_I_rank1, Nested_J_rank1)
error = tl.norm(f1_tensor - recon_f1) / tl.norm(f1_tensor)
print(f"Relative error of f1 QTT at r_max = {r_max}: {error}")

# TCI-2site of f2
r_max = 5
interp_I_f2, interp_J_f2, TTRank_f2, recon_f2 = TCI_2site(f2_tensor, 0, r_max, Nested_I_rank1, Nested_J_rank1)
error = tl.norm(f2_tensor - recon_f2) / tl.norm(f2_tensor)
print(f"Relative error of f2 QTT at r_max = {r_max}: {error}")

# TCI-2site of g
r_max = 7
interp_I_g, interp_J_g, TTRank_g, recon_g = TCI_2site(g_tensor, 0, r_max, Nested_I_rank1, Nested_J_rank1)
error = tl.norm(g_tensor - recon_g) / tl.norm(g_tensor)
print(f"Relative error of g QTT at r_max = {r_max}: {error}")

# Assemble f1
TT_cross_f1 = cross_core_interp_assemble(f1_tensor, interp_I_f1, interp_J_f1, TTRank_f1)
TT_cores_f1 = cross_inv_merge(TT_cross_f1, dim, 1)

# Assemble f2
TT_cross_f2 = cross_core_interp_assemble(f2_tensor, interp_I_f2, interp_J_f2, TTRank_f2)
TT_cores_f2 = cross_inv_merge(TT_cross_f2, dim, 1)

''' === Limitation of RED - Reconstruction, Evaluation, Decomposition === '''
def red_test():
    print("=== Limitation of RED - Reconstruction, Evaluation, Decomposition ===")

    rerr_f1 = tl.norm(f1_tensor - recon_f1) / tl.norm(f1_tensor)
    print(f"||f1 TCI - f1 tensor|| / ||f1 tensor|| = {rerr_f1}")   
    
    rerr_f2 = tl.norm(f2_tensor - recon_f2) / tl.norm(f2_tensor)
    print(f"||f2 TCI - f2 tensor|| / ||f2 tensor|| = {rerr_f2}")

    red_f1f2 = recon_f1 * recon_f2
    rerr_f1f2 = tl.norm(red_f1f2- g_tensor) / tl.norm(g_tensor)
    print(f"||f1 TCI x f2 TCI - f1 tensor x f2 tensor|| / ||f1 tensor x f2 tensor|| = {rerr_f1f2}")
    
    for r_max in range(2,9):
        interp_I_f1f2, interp_J_f1f2, TTRank_f1f2, recon_f1f2 = TCI_2site(red_f1f2, 0, r_max, Nested_I_rank1, Nested_J_rank1)
        error = tl.norm(recon_f1f2 - g_tensor) / tl.norm(g_tensor)
        print(f"Relative error of g RED at r_max = {r_max}: {error}")    
    return

red_test()
pass

''' === Test the idea of hierarchical integral === '''
def hInt_firstTry():
    # Let's first say how hInt can give optimal I pivots from left side compared with direct TCI
    interp_I_f1_copy = interp_I_f1.copy()
    interp_I_f2_copy = interp_I_f2.copy()
    interp_J_new = {}
    
    # Maximal TT-Rank for target TT after product
    contract_core_number = 4
    max_rank = 7

    # Hierarchical Integral Iteration
    passed_core_number = 0
    r_ = 1
    Zj_list = []
    TTRank_new = [1]
    while passed_core_number < dim-1:
        # Do we need to integrate
        temp = dim - passed_core_number - contract_core_number
        integral_number = temp if temp > 0 else 0 
        print(f"# integral: {integral_number}, # contraction {contract_core_number}, # passed core {passed_core_number}.")

        # Partially contract TT cores of f1 and f2
        f1_contraction = single_core_interp_assemble(f1_tensor, interp_I_f1_copy, interp_J_f1, TTRank_f1, passed_core_number)
        f2_contraction = single_core_interp_assemble(f2_tensor, interp_I_f2_copy, interp_J_f2, TTRank_f2, passed_core_number)
        for i in range(1, contract_core_number):
            f1_core = single_core_interp_assemble(f1_tensor, interp_I_f1_copy, interp_J_f1, TTRank_f1, passed_core_number+i)
            f2_core = single_core_interp_assemble(f2_tensor, interp_I_f2_copy, interp_J_f2, TTRank_f2, passed_core_number+i)
            f1_contraction = np.tensordot(f1_contraction, f1_core, axes=([len(f1_contraction.shape)-1],[0]))
            f2_contraction = np.tensordot(f2_contraction, f2_core, axes=([len(f2_contraction.shape)-1],[0]))
        
        if integral_number > 0:
            # Partially integrate f1 and f2 via QTT
            integral_qtci_f1 = integral_qtt(TT_cores_f1, integral_number, 1)
            integral_qtci_f2 = integral_qtt(TT_cores_f2, integral_number, 1)
            f1_contraction = f1_contraction @ integral_qtci_f1.reshape(-1,1)
            f2_contraction = f2_contraction @ integral_qtci_f2.reshape(-1,1) 

        # Hadamard product
        TTint_contract_f1f2 = f1_contraction * f2_contraction
            
        # New pivot selection
        shape_contract_t = TTint_contract_f1f2.shape 
        mat_row = ma.prod(shape_contract_t[0:2])
        mat_col = ma.prod(shape_contract_t[2:])
        TTint_matrix = tl.reshape(TTint_contract_f1f2, [mat_row, mat_col])    
        #_, diag, _, _, _, pr, pc, _ = prrldu(TTint_matrix, eps, 5)
        r_subset, c_subset, cross_inv, cross, rank, pr, pc = cur_prrldu(TTint_matrix, 0, max_rank)
        Z = c_subset @ cross_inv 
        Z_core = tl.reshape(c_subset, [r_, 2, rank])
        Zj_list.append(Z_core)
        
        r_ = rank
        TTRank_new.append(rank)
        TTRank_f1[passed_core_number+1] = rank
        TTRank_f2[passed_core_number+1] = rank
        print(f"Tensor contraction {TTint_contract_f1f2.shape} -> Matrix {TTint_matrix.shape} -> rank revealing -> Z core {Z_core.shape}")

        # Mapping between r/c selection and tensor index pivots
        if passed_core_number == 0:
            interp_I_f1_copy[passed_core_number+1] = np.array(pr).reshape(-1, 1)
            interp_I_f2_copy[passed_core_number+1] = np.array(pr).reshape(-1, 1)
        else:
            I = np.empty([rank, passed_core_number+1])
            prev_I = interp_I_f1_copy[passed_core_number]
            for j in range(rank):
                p_I_idx = pr[j] // 2
                c_i_idx = pr[j] % 2
                I[j,0:passed_core_number] = prev_I[p_I_idx]
                I[j,passed_core_number] = c_i_idx
            interp_I_f1_copy[passed_core_number+1] = I
            interp_I_f2_copy[passed_core_number+1] = I        

        if passed_core_number == dim - 2:
            interp_J_new[dim] = np.array(pc).reshape(-1,1) 

        # Update TTRank
        TTRank_f1[passed_core_number] = interp_I_f1_copy[passed_core_number+1].shape[0]
        TTRank_f2[passed_core_number] = interp_I_f2_copy[passed_core_number+1].shape[0]
        passed_core_number += 1
        if integral_number == 0:
            contract_core_number = dim - passed_core_number
    TTRank_new.append(1)
    
    # Now I set is done. Let's try to figure out the J direction\
    # !! To be discussed here... Use 1-site TCI?
    pass
    for i in range(dim-2, 0, -1):
        ccore = Zj_list[i] 
        cshape = ccore.shape
        zmat = tl.reshape(ccore, [cshape[0], cshape[1] * cshape[2]], order='F')
        _, _, _, _, _, rps, cps, _ = prrldu(zmat, 0, cshape[0]) 
        curr_dim = cshape[1]
        prev_J = interp_J_new[i+2]
        J = np.empty([cshape[0], dim-i])
        for j in range(cshape[0]):
            p_J_idx = cps[j] // curr_dim
            c_J_idx = cps[j] % curr_dim
            J[j,1:] = prev_J[p_J_idx]
            J[j,0] = c_J_idx
        interp_J_new[i+1] = J

    return interp_I_f1_copy, interp_J_new, TTRank_new

# Get new g's I, J sets from f1 TCI and f2 TCI via the integral method
interp_I_g_new, interp_J_g_new, TTRank_g_new = hInt_firstTry()

TT_cross_g_TCI = cross_core_interp_assemble(g_tensor, interp_I_g, interp_J_g, TTRank_g)
TT_cores_g_TCI = cross_inv_merge(TT_cross_g_TCI, dim, 1)
error = tl.norm(g_tensor - tl.tt_to_tensor(TT_cores_g_TCI)) / tl.norm(g_tensor)
print(f"Relative error of g QTT at r_max = {r_max}: {error}")

TT_cross_g_new = cross_core_interp_assemble(g_tensor, interp_I_g_new, interp_J_g_new, TTRank_g_new)
TT_cores_g_new = cross_inv_merge(TT_cross_g_new, dim, 1)
error = tl.norm(g_tensor - tl.tt_to_tensor(TT_cores_g_new)) / tl.norm(g_tensor)
print(f"Relative error of g new at r_max = {r_max}: {error}")
pass


# Plot numerical results
max_rank_selection = [2,3,4,5,6,7,8]
rel_error_TCI_g = [0.518606738251041, 0.03929485422630780, 0.005080629328349756, 0.0022383885452628596, 0.0001335697037330801, 6.460937515610282e-05, 5.4462581444022203e-05]
rel_error_INT_g = [0.387611297071045, 0.03474862630575235, 0.007601452070678471, 0.0021256243069158304, 0.0002251037306136618, 2.463364284198022e-05, 0.013284752015896135]
max_rank_union = [4, 7]
rel_error_UNI_g = [0.016386594417341145, 0.0001748485620257831]

plt.figure()
plt.scatter(max_rank_union, rel_error_UNI_g, label="Union-prrlu-f1f2", marker='o',color='orange')
plt.scatter(max_rank_selection, rel_error_INT_g, label="Integral-Contraction-f1f2", marker='s',color='green')
plt.scatter(max_rank_selection, rel_error_TCI_g, label="TCI-prrlu-2site (RED)", marker='x',color='red')
plt.annotate("outlier here, due to\nunstable matrix inversion", 
             xy=(8, rel_error_INT_g[6]),  # Point to annotate (rank=8, corresponding error)
             xytext=(7.2, 0.05),          # Position of the text
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=10,
             color='red',
             ha='center')
plt.yscale("log")
plt.grid()
plt.legend()
plt.xlabel("TT-Rank")
plt.ylabel("Relative difference from the real f1xf2")
plt.savefig("relerr_UNI_vs_TCI_vs_INT.png")
