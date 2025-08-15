import time as tm
import numpy as np
import math as ma
import tensorly as tl

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MLA-Toolkit', 'py'))

from tensor_cross import single_core_interp_assemble, cross_core_interp_assemble, cross_inv_merge, cross_inv_merge_stable, coreinv_qr, slice_first_modes, slice_last_modes
from qtt import Qintegral_TT, adj_ttcore_contract
from interpolation import cur_prrldu
from rank_revealing import prrldu

# TODO...
# 1. Performance (runtime) profiling
# 2. New assemble of f1/f2 TT-core need to trust interpolation
# 3. Check the repeated operations

# Element-wise correctness check of partial contraction
def partial_recon_ele_check(f_tensor, interp_I, interp_J, offset, contract_core_number, contraction):
    dim = len(f_tensor.shape)  # Tensor dimension
    I_pos = offset
    J_pos = offset + contract_core_number + 1
    I_set = interp_I[I_pos]
    J_set = interp_J[J_pos]

    max_rel_err = 0
    if I_pos != 0 and J_pos != [dim + 1]:
        for i in range(len(I_set)):
            tensor_slice_i = slice_first_modes(f_tensor, I_set[i].astype(int).tolist())
            contract_slice_i = slice_first_modes(contraction, [i])
            for j in range(len(J_set)):
                tensor_slice_ij = slice_last_modes(tensor_slice_i, J_set[j].astype(int).tolist())
                contract_slice_ij = slice_last_modes(contract_slice_i, [j])
                rel_err = tl.norm(tensor_slice_ij - contract_slice_ij) / tl.norm(tensor_slice_ij)
                max_rel_err = max(max_rel_err, rel_err)
    else:
        if I_pos == 0:
            for j in range(len(J_set)):
                tensor_slice_j = slice_last_modes(f_tensor, J_set[j].astype(int).tolist())
                contract_slice_j = slice_last_modes(contraction, [j])
                rel_err = tl.norm(tensor_slice_j - contract_slice_j) / tl.norm(tensor_slice_j)
                max_rel_err = max(max_rel_err, rel_err)
        if J_pos == dim + 1:
            for i in range(len(I_set)):
                tensor_slice_i = slice_first_modes(f_tensor, I_set[i].astype(int).tolist())
                contract_slice_i = slice_first_modes(contraction, [i])
                rel_err = tl.norm(tensor_slice_i - contract_slice_i) / tl.norm(tensor_slice_i)
                max_rel_err = max(max_rel_err, rel_err)
    
    return max_rel_err


# QTTM algorithm with access of f1/f2 functions. Partial integral and contraction method.
def QTTM_INTCONT(f1_tensor, interp_I_f1, interp_J_f1, TTRank_f1, pr_set_f1,
                 f2_tensor, interp_I_f2, interp_J_f2, TTRank_f2, pr_set_f2,
                 contract_core_number, max_rank, eps, verbose = 1):
    start_time_QTTM = tm.time()

    # Assemble QTT-cores of f1 and f2 via the TCI interpolation
    TT_cross_f1 = cross_core_interp_assemble(f1_tensor, interp_I_f1, interp_J_f1, TTRank_f1)
    TT_cross_f2 = cross_core_interp_assemble(f2_tensor, interp_I_f2, interp_J_f2, TTRank_f2)
    TT_cores_f1 = cross_inv_merge_stable(TT_cross_f1, pr_set_f1)
    TT_cores_f2 = cross_inv_merge_stable(TT_cross_f2, pr_set_f2) 
    if verbose == 1:
        f1_TTerr = tl.norm(f1_tensor - tl.tt_to_tensor(TT_cores_f1)) / tl.norm(f1_tensor)
        f2_TTerr = tl.norm(f2_tensor - tl.tt_to_tensor(TT_cores_f2)) / tl.norm(f2_tensor)
        print(f"QTTM Verbose: Relative error of f1 QTT: {f1_TTerr}, f2 QTT: {f2_TTerr}")

    # Preparation for g's I, J basis
    dim = len(TTRank_f1) - 1  # Tensor dimension
    interp_I_f1_gBasis = interp_I_f1.copy()
    interp_I_f2_gBasis = interp_I_f2.copy()
    interp_J_g_new = {}
    
    # Integral cache
    f1_int_tt = Qintegral_TT(TT_cores_f1)
    f2_int_tt = Qintegral_TT(TT_cores_f2)

    # Preliminary: Before iteration...
    passed_core_number = 0  # Iteration number
    r_g = 1                 # Next TT-Rank of g 
    Zj_list = []            # Interpolation Zj cores
    Csubset_list = []       # Skeleton C subset cores
    TTRank_new = [1]        # TT-rank list of g
    
    # Hierarchical Integral-Contraction Iteration
    while passed_core_number < dim-1:
        # Check if we need to integrate the last cores
        temp = dim - passed_core_number - contract_core_number
        integral_number = temp if temp > 0 else 0 
        print(f"# integral: {integral_number}, # contraction {contract_core_number}, # passed core {passed_core_number}.")

        # Make sure that I interpolation in g basis are unified
        assert np.array_equal(interp_I_f1_gBasis[passed_core_number], interp_I_f2_gBasis[passed_core_number]), f"f1 and f2 interpolation g basis do not match at position {passed_core_number}"

        # Partial reconstruction   
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
        r_subset, c_subset, _, _, rank, pr, pc = cur_prrldu(TTint_matrix, eps, max_rank)
        pr = pr[0: rank]
        pc = pc[0: rank]
        
        # One-side ID: Z Interpolation core
        c_subset_3d = tl.reshape(c_subset, [r_g, 2, rank])
        Csubset_list.append(c_subset_3d)
        Z_core = coreinv_qr(c_subset_3d, pr)
        Zj_list.append(Z_core)
        if passed_core_number == dim-2:
            last_core = tl.reshape(r_subset, [rank, 2, 1])
            Zj_list.append(last_core)
        
        r_g = rank
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

    end_time_QTTM = tm.time()
    print(f"QTTM INTCONT runti me: {end_time_QTTM - start_time_QTTM:.4f} seconds")
    return interp_I_f1_gBasis, interp_J_g_new, TTRank_new, Zj_list


# QTTM algorithm with access of f1/f2 functions. Partial integral and contraction method. In this version we don't have evaluation of f1 and f2
def QTTM_INTCONT_NOEVAL(TT_cores_f1, interp_I_f1, TTRank_f1,
                        TT_cores_f2, interp_I_f2, TTRank_f2,
                        contract_core_number, max_rank, eps):
    start_time_QTTM = tm.time()

    # Preparation for g's I, J basis
    dim = len(TTRank_f1) - 1  # Tensor dimension
    TTRank_f1 = TTRank_f1.copy()
    TTRank_f2 = TTRank_f2.copy()
    interp_I_f1_gBasis = interp_I_f1.copy()
    interp_I_f2_gBasis = interp_I_f2.copy()
    interp_J_g_new = {}
    
    # Integral cache
    f1_int_tt = Qintegral_TT(TT_cores_f1)
    f2_int_tt = Qintegral_TT(TT_cores_f2)

    # Preliminary: Before iteration...
    passed_core_number = 0  # Iteration number
    r_g = 1                 # Next TT-Rank of g 
    Zj_list = []            # Interpolation Zj cores
    Csubset_list = []       # Skeleton C subset cores
    TTRank_new = [1]        # TT-rank list of g
    
    # Hierarchical Integral-Contraction Iteration
    while passed_core_number < dim-1:
        # Check if we need to integrate the last cores
        temp = dim - passed_core_number - contract_core_number
        integral_number = temp if temp > 0 else 0 
        print(f"# integral: {integral_number}, # contraction {contract_core_number}, # passed core {passed_core_number}.")

        # Make sure that I interpolation in g basis are unified
        assert np.array_equal(interp_I_f1_gBasis[passed_core_number], interp_I_f2_gBasis[passed_core_number]), f"f1 and f2 interpolation g basis do not match at position {passed_core_number}"
        if passed_core_number > 0:
            # Cache contraction to approximate values not included in interpolation
            I_gbasis_curr = interp_I_f1_gBasis[passed_core_number]
            I_gbasis_prev = interp_I_f1_gBasis[passed_core_number - 1]
            cache_contract_f1 = adj_ttcore_contract(TT_cores_f1[passed_core_number-1], TT_cores_f1[passed_core_number])
            cache_contract_f2 = adj_ttcore_contract(TT_cores_f2[passed_core_number-1], TT_cores_f2[passed_core_number])   
            
            curr_core_f1 = np.empty([len(I_gbasis_curr), 2, len(interp_I_f1_gBasis[passed_core_number + 1])])
            curr_core_f2 = np.empty([len(I_gbasis_curr), 2, len(interp_I_f2_gBasis[passed_core_number + 1])])

            # Get approximated TT-cores interpolated by new g basis
            for i in range(len(I_gbasis_curr)):
                curr_pivot_i = I_gbasis_curr[i]
                last_pivot = curr_pivot_i[-1].astype(int)    
                arg_idx = 0
                if passed_core_number > 1:
                    prev_pivot = curr_pivot_i[0:-1]
                    matches = np.all(I_gbasis_prev == prev_pivot, axis = 1)
                    arg_idx = np.where(matches)[0][0]
                curr_core_f1[i, :, :] = cache_contract_f1[arg_idx, last_pivot, :, :]
                curr_core_f2[i, :, :] = cache_contract_f2[arg_idx, last_pivot, :, :]
            
            TT_cores_f1[passed_core_number] = curr_core_f1
            TT_cores_f2[passed_core_number] = curr_core_f2
            
        # Partial contraction of f1 and f2
        f1_contract = TT_cores_f1[passed_core_number]
        f2_contract = TT_cores_f2[passed_core_number]
        for i in range(1, contract_core_number):
            f1_contract = np.tensordot(f1_contract, TT_cores_f1[passed_core_number + i], axes=([len(f1_contract.shape)-1],[0]))
            f2_contract = np.tensordot(f2_contract, TT_cores_f2[passed_core_number + i], axes=([len(f2_contract.shape)-1],[0]))

        # Partially integrate f1 and f2 via QTT
        if integral_number > 0:
            int_f1 = f1_int_tt[-1]
            int_f2 = f2_int_tt[-1]
            for i in range(dim - 2, dim - integral_number - 1, -1):
                sub_int_f1 = f1_int_tt[i]
                sub_int_f2 = f2_int_tt[i]
                int_f1 = sub_int_f1 @ int_f1
                int_f2 = sub_int_f2 @ int_f2
            f1_contract = f1_contract @ int_f1.reshape(-1,1)
            f2_contract = f2_contract @ int_f2.reshape(-1,1)

        # Hadamard product
        TTint_contract_f1f2 = f1_contract * f2_contract
            
        # New pivot selection
        shape_contract_t = TTint_contract_f1f2.shape 
        mat_row = ma.prod(shape_contract_t[0:2])
        mat_col = ma.prod(shape_contract_t[2:])
        TTint_matrix = tl.reshape(TTint_contract_f1f2, [mat_row, mat_col])    
        r_subset, c_subset, _, rank, pr, pc = cur_prrldu(TTint_matrix, eps, max_rank)
        pr = pr[0: rank]
        pc = pc[0: rank]
        
        # One-side ID: Z Interpolation core
        c_subset_3d = tl.reshape(c_subset, [r_g, 2, rank])
        Csubset_list.append(c_subset_3d)
        Z_core = coreinv_qr(c_subset_3d, pr)
        Zj_list.append(Z_core)
        if passed_core_number == dim-2:
            last_core = tl.reshape(r_subset, [rank, 2, 1])
            Zj_list.append(last_core)
        
        r_g = rank
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

    end_time_QTTM = tm.time()
    print(f"QTTM INTCONT runti me: {end_time_QTTM - start_time_QTTM:.4f} seconds")
    return interp_I_f1_gBasis, interp_J_g_new, TTRank_new, Zj_list