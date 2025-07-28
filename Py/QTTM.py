import numpy as np
import math as ma
import tensorly as tl
from tensor_cross import single_core_interp_assemble, cross_core_interp_assemble, cross_inv_merge
from QTT import integral_qtt, Qintegral_TT
from interpolation import cur_prrldu
from rank_revealing import prrldu

# TODO...
# 1. Performance (runtime) profiling
# 2. New assemble of f1/f2 TT-core need to trust interpolation
# 3. Check the repeated operations
def QTTM_INTCONT(f1_tensor, interp_I_f1, interp_J_f1, TTRank_f1,
                 f2_tensor, interp_I_f2, interp_J_f2, TTRank_f2,
                 contract_core_number, max_rank, dim):
    # Assemble QTT-cores of f1 and f2 via the TCI interpolation
    # This part seems not that stable so far
    TT_cross_f1 = cross_core_interp_assemble(f1_tensor, interp_I_f1, interp_J_f1, TTRank_f1)
    TT_cross_f2 = cross_core_interp_assemble(f2_tensor, interp_I_f2, interp_J_f2, TTRank_f2)
    TT_cores_f1 = cross_inv_merge(TT_cross_f1, dim, 1)  # TODO: The inverse is unstable -> QR?
    TT_cores_f2 = cross_inv_merge(TT_cross_f2, dim, 1)       

    interp_I_f1_gBasis = interp_I_f1.copy()
    interp_I_f2_gBasis = interp_I_f2.copy()
    interp_J_g_new = {}
    
    # Integral cache
    f1_int_tt = Qintegral_TT(TT_cores_f1)
    f2_int_tt = Qintegral_TT(TT_cores_f2)

    # Hierarchical Integral-Contraction Iteration
    passed_core_number = 0
    r_ = 1
    Zj_list = []
    Csubset_list = []
    TTRank_new = [1]
    while passed_core_number < dim-1:
        # Do we need to integrate
        temp = dim - passed_core_number - contract_core_number
        integral_number = temp if temp > 0 else 0 
        print(f"# integral: {integral_number}, # contraction {contract_core_number}, # passed core {passed_core_number}.")

        # Partially contract TT cores of f1 and f2
        # TODO: The current problem is now assembling new TT-cores of f1/f2 using g basis still requires original f1/f2 tensor. We need to trust interpolation 
        f1_contraction = single_core_interp_assemble(f1_tensor, interp_I_f1_gBasis, interp_J_f1, TTRank_f1, passed_core_number)
        f2_contraction = single_core_interp_assemble(f2_tensor, interp_I_f2_gBasis, interp_J_f2, TTRank_f2, passed_core_number)
        for i in range(1, contract_core_number):
            f1_core = single_core_interp_assemble(f1_tensor, interp_I_f1_gBasis, interp_J_f1, TTRank_f1, passed_core_number+i)
            f2_core = single_core_interp_assemble(f2_tensor, interp_I_f2_gBasis, interp_J_f2, TTRank_f2, passed_core_number+i)
            f1_contraction = np.tensordot(f1_contraction, f1_core, axes=([len(f1_contraction.shape)-1],[0]))
            f2_contraction = np.tensordot(f2_contraction, f2_core, axes=([len(f2_contraction.shape)-1],[0]))
        
        # Partially integrate f1 and f2 via QTT
        if integral_number > 0:
            int_f1 = f1_int_tt[-1]
            int_f2 = f2_int_tt[-1]
            for i in range(dim - 2, dim - integral_number - 1, -1):
                sub_int_f1 = f1_int_tt[i]
                sub_int_f2 = f2_int_tt[i]
                int_f1 = sub_int_f1 @ int_f1
                int_f2 = sub_int_f2 @ int_f2
            f1_contraction = f1_contraction @ int_f1.reshape(-1,1)
            f2_contraction = f2_contraction @ int_f2.reshape(-1,1)

        # Hadamard product
        TTint_contract_f1f2 = f1_contraction * f2_contraction
            
        # New pivot selection
        shape_contract_t = TTint_contract_f1f2.shape 
        mat_row = ma.prod(shape_contract_t[0:2])
        mat_col = ma.prod(shape_contract_t[2:])
        TTint_matrix = tl.reshape(TTint_contract_f1f2, [mat_row, mat_col])    
        #_, diag, _, _, _, pr, pc, _ = prrldu(TTint_matrix, eps, 5)
        r_subset, c_subset, cross_inv, cross, rank, pr, pc = cur_prrldu(TTint_matrix, 0, max_rank)
        Csubset_list.append(tl.reshape(c_subset, [r_, 2, rank]))
        Z = c_subset @ cross_inv 
        Z_core = tl.reshape(Z, [r_, 2, rank])
        Zj_list.append(Z_core)
        if passed_core_number == dim-2:
            last_core = tl.reshape(r_subset, [rank, 2, 1])
            Zj_list.append(last_core)
        
        r_ = rank
        TTRank_new.append(rank)
        TTRank_f1[passed_core_number+1] = rank
        TTRank_f2[passed_core_number+1] = rank
        print(f"Tensor contraction {TTint_contract_f1f2.shape} -> Matrix {TTint_matrix.shape} -> rank revealing -> Z core {Z_core.shape}")

        # Mapping between r/c selection and tensor index pivots
        if passed_core_number == 0:
            interp_I_f1_gBasis[passed_core_number+1] = np.array(pr).reshape(-1, 1)
            interp_I_f2_gBasis[passed_core_number+1] = np.array(pr).reshape(-1, 1)
        else:
            I = np.empty([rank, passed_core_number+1])
            prev_I = interp_I_f1_gBasis[passed_core_number]
            for j in range(rank):
                p_I_idx = pr[j] // 2
                c_i_idx = pr[j] % 2
                I[j,0:passed_core_number] = prev_I[p_I_idx]
                I[j,passed_core_number] = c_i_idx
            interp_I_f1_gBasis[passed_core_number+1] = I
            interp_I_f2_gBasis[passed_core_number+1] = I        

        if passed_core_number == dim - 2:
            interp_J_g_new[dim] = np.array(pc).reshape(-1,1) 

        # Update TTRank
        TTRank_f1[passed_core_number] = interp_I_f1_gBasis[passed_core_number+1].shape[0]
        TTRank_f2[passed_core_number] = interp_I_f2_gBasis[passed_core_number+1].shape[0]
        passed_core_number += 1
        if integral_number == 0:
            contract_core_number = dim - passed_core_number
    TTRank_new.append(1)
    
    # Now I set is done. Let's try to figure out the J direction\
    # !! To be discussed here... Use 1-site TCI?
    pass
    for i in range(dim-2, 0, -1):
        ccore = Csubset_list[i] 
        cshape = ccore.shape
        zmat = tl.reshape(ccore, [cshape[0], cshape[1] * cshape[2]], order='F')
        _, _, _, _, _, rps, cps, _ = prrldu(zmat, 0, cshape[0]) 
        curr_dim = cshape[1]
        prev_J = interp_J_g_new[i+2]
        J = np.empty([cshape[0], dim-i])
        for j in range(cshape[0]):
            p_J_idx = cps[j] // curr_dim
            c_J_idx = cps[j] % curr_dim
            J[j,1:] = prev_J[p_J_idx]
            J[j,0] = c_J_idx
        interp_J_g_new[i+1] = J

    return interp_I_f1_gBasis, interp_J_g_new, TTRank_new, Zj_list