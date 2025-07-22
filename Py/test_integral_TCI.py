import numpy as np
import tensorly as tl
import math as ma
import matplotlib.pyplot as plt

from rank_revealing import prrldu
from QTT import populate_tensor_fromfunction, union_rows_bounded, scatter_plot_f1f2, integral_qtt, value_query_QTT
from tensor_cross import TT_CUR_L2R, cross_core_interp_assemble, TCI_2site, cross_inv_merge, TCI_union_two, single_core_interp_assemble

''' === Quantics representation construction === '''
# Quantics construction
quantic_repres = lambda x1,x2,x3,x4,x5,x6,x7,x8,x9,x10: x1/2 + x2/(2**2) + x3/(2**3) + x4/(2**4) + x5/(2**5) + x6/(2**6) + x7/(2**7) + x8/(2**8) + x9/(2**9) + x10/(2**10)
func1 = lambda t: 1.2 * t ** 4 - 0.2 * np.sqrt(t) - 1 + 0.6 * np.sin(7.3 * np.pi * t)  #t ** 5 - 3 * t ** 3 + 10 * t -6 #5 * np.sin(-2 * np.pi * t) - 3 * np.exp(t)
func2 = lambda t: -1.1 * t ** 7 - 12 + np.exp(3.1*t) - 0.81 * np.cos(6 * np.pi * t) - 2 * t ** 2 + 4 + np.tan(t)  #-10 * np.exp(-(t - 1) * (t - 1) / 2) - 2 * t ** 3 + 4 g_func = lambda t: func1(t) * func2(t)
#func1 = lambda t: 1024 * t**8 * (1-t)**8 * (2*t - 1)**2
#func2 = lambda t: 4096 * t**10 * (1-t)**10 * (1 - 6*t + 6*t**2)**2
shape = (2,2,2,2,2,2,2,2,2,2)
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
r_max = 5
interp_I_g, interp_J_g, TTRank_g, recon_g = TCI_2site(g_tensor, 1e-10, r_max, Nested_I_rank1, Nested_J_rank1)
error = tl.norm(g_tensor - recon_g) / tl.norm(g_tensor)
print(f"Relative error of g QTT at r_max = {r_max}: {error}")

# Assemble f1
TT_cross_f1 = cross_core_interp_assemble(f1_tensor, interp_I_f1, interp_J_f1, TTRank_f1)
TT_cores_f1 = cross_inv_merge(TT_cross_f1, dim, 1)

# Assemble f2
TT_cross_f2 = cross_core_interp_assemble(f2_tensor, interp_I_f2, interp_J_f2, TTRank_f2)
TT_cores_f2 = cross_inv_merge(TT_cross_f2, dim, 1)

# Maximal TT-Rank for target TT after product
contract_core_number = 4

''' === Test the idea of hierarchical integral === '''
def hInt_firstTry():
    # Let's first say how hInt can give optimal I pivots from left side compared with direct TCI
    interp_I_f1_copy = interp_I_f1.copy()
    interp_I_f2_copy = interp_I_f2.copy()
    
    # Hierarchical Integral Iteration
    passed_core_number = 0
    while passed_core_number < dim:
        # Do we need to integrate
        integral_number = dim - passed_core_number - contract_core_number
        print(f"# integral: {integral_number}, # contraction {contract_core_number}, # passed core {passed_core_number}.")

        if integral_number > 0:
            # Partially integrate f1 and f2 via QTT
            integral_qtci_f1 = integral_qtt(TT_cores_f1, integral_number, 1)
            integral_qtci_f2 = integral_qtt(TT_cores_f2, integral_number, 1)
            
            # Partially contract TT cores of f1 and f2
            f1_contraction = single_core_interp_assemble(f1_tensor, interp_I_f1_copy, interp_J_f1, TTRank_f1, passed_core_number)
            f2_contraction = single_core_interp_assemble(f2_tensor, interp_I_f2_copy, interp_J_f2, TTRank_f2, passed_core_number)
            for i in range(1, contract_core_number):
                f1_core = single_core_interp_assemble(f1_tensor, interp_I_f1_copy, interp_J_f1, TTRank_f1, passed_core_number+i)
                f2_core = single_core_interp_assemble(f2_tensor, interp_I_f2_copy, interp_J_f2, TTRank_f2, passed_core_number+i)
                f1_contraction = np.tensordot(f1_contraction, f1_core, axes=([len(f1_contraction.shape)-1],[0]))
                f2_contraction = np.tensordot(f2_contraction, f2_core, axes=([len(f2_contraction.shape)-1],[0]))
            f1_contraction = f1_contraction @ integral_qtci_f1.reshape(-1,1)
            f2_contraction = f2_contraction @ integral_qtci_f2.reshape(-1,1) 

            # Hadamard product
            TTint_contract_f1f2 = f1_contraction * f2_contraction
            
            # New pivot selection
            shape_contract_t = TTint_contract_f1f2.shape 
            mat_row = ma.prod(shape_contract_t[0:2])
            mat_col = ma.prod(shape_contract_t[2:])
            TTint_matrix = tl.reshape(TTint_contract_f1f2, [mat_row, mat_col])    
            _, diag, _, _, _, pr, pc, _ = prrldu(TTint_matrix, eps, 5)
            rank = len(diag)

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

            # REMEMBER: update TTRank
            TTRank_f1[passed_core_number] = interp_I_f1_copy[passed_core_number+1].shape[0]
            TTRank_f2[passed_core_number] = interp_I_f2_copy[passed_core_number+1].shape[0]

        else:
            pass       

        passed_core_number += 1

    return


hInt_firstTry()





