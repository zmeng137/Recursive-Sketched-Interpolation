import os
import sys
import time as tm
import numpy as np
import math as ma
import tensorly as tl

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MLA-Toolkit', 'py'))
from tci import single_core_interp_assemble, cross_core_interp_assemble, cross_inv_merge_stable, coreinv_qr, coreinv_lu, slice_first_modes, slice_last_modes
from interpolation import cur_prrldu, interpolative_prrldu
from rank_revealing import prrldu
from qtt import Qintegral_TT, adj_ttcore_contract, qtt_sketching_cache

# TODO...
# 1. Performance (runtime) profiling
# 2. New assemble of f1/f2 TT-core need to trust interpolation
# 3. Check the repeated operations


# QTTM-RIC: (Q)uantics (T)ensor (T)rain (M)ultiplication via (R)andomized (I)nterpolation and (C)ontraction.
def qttm_ric(TT_cores_f1, interp_I_f1, TTRank_f1,
             TT_cores_f2, interp_I_f2, TTRank_f2,
             contract_core_number, max_rank, eps,
             randomFlag, seed, skLayer):
    # Preparation for g's I, J basis
    dim = len(TTRank_f1) - 1  # Tensor dimension
    TTRank_f1 = TTRank_f1.copy()
    TTRank_f2 = TTRank_f2.copy()
    interp_I_f1_gBasis = interp_I_f1.copy()
    interp_I_f2_gBasis = interp_I_f2.copy()
    interp_J_g_new = {}
    start_time_QTTM = tm.time()
    
    # Randomized sketching cache
    start_time_sketch_cache = tm.time()
    f1_sk_tt, skLayer = qtt_sketching_cache(TT_cores_f1, randomFlag, seed, skLayer)
    f2_sk_tt, skLayer = qtt_sketching_cache(TT_cores_f2, randomFlag, seed, skLayer)
    end_time_sketch_cache = tm.time()
    elapsed_time_skcache = (end_time_sketch_cache - start_time_sketch_cache)
    
    # Preliminary: Before iteration...
    passed_core_number = 0  # Iteration number
    r_g = 1                 # Next TT-Rank of g 
    Zj_list = []            # Interpolation Zj cores
    TTRank_new = [1]        # TT-rank list of g
    
    # Elapsed time
    elapsed_time_sketching = 0
    elapsed_time_contraction = 0
    elapsed_time_decomp = 0
    elapsed_time_product = 0
    elapsed_time_basis = 0

    # Hierarchical Integral-Contraction Iteration
    while passed_core_number < dim-1:
        # Check if we need to integrate the last cores
        temp = dim - passed_core_number - contract_core_number
        sketch_number = temp if temp > 0 else 0 

        print(f"# sketch: {sketch_number}, # contraction {contract_core_number}, # passed core {passed_core_number}.")

        # Make sure that I interpolation in g basis are unified
        assert np.array_equal(interp_I_f1_gBasis[passed_core_number], interp_I_f2_gBasis[passed_core_number]), f"f1 and f2 interpolation g basis do not match at position {passed_core_number}"
        start_contraction = tm.time()
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
        end_contraction = tm.time()
        elapsed_time_contraction += (end_contraction - start_contraction)

        # Randomized sketching of f1/f2's QTT
        start_sketching = tm.time()
        if sketch_number > 0:
            skCore_f1 = np.zeros([f1_contract.shape[-1], skLayer])
            skCore_f2 = np.zeros([f2_contract.shape[-1], skLayer])
            for l in range(skLayer):
                sk_f1 = f1_sk_tt[l][-1]
                sk_f2 = f2_sk_tt[l][-1]
                for i in range(dim - 2, dim - sketch_number - 1, -1):
                    sub_int_f1 = f1_sk_tt[l][i]
                    sub_int_f2 = f2_sk_tt[l][i]
                    sk_f1 = sub_int_f1 @ sk_f1
                    sk_f2 = sub_int_f2 @ sk_f2
                skCore_f1[:, l] = np.squeeze(sk_f1, axis=-1)
                skCore_f2[:, l] = np.squeeze(sk_f2, axis=-1)
            f1_contract = f1_contract @ skCore_f1
            f2_contract = f2_contract @ skCore_f2
        end_sketching = tm.time()
        elapsed_time_sketching += (end_sketching - start_sketching)

        # Hadamard product
        start_product = tm.time()
        TTint_contract_f1f2 = f1_contract * f2_contract
        end_product = tm.time()
        elapsed_time_product += (end_product - start_product)

        # New pivot selection (decomposition)
        start_decomp = tm.time()
        
        shape_contract_t = TTint_contract_f1f2.shape 
        mat_row = ma.prod(shape_contract_t[0:2])
        mat_col = ma.prod(shape_contract_t[2:])
        skTT_matrix = tl.reshape(TTint_contract_f1f2, [mat_row, mat_col])    
        
        # One-side ID: Z Interpolation core
        Ctrans, Ztrans, pr, _ = interpolative_prrldu(skTT_matrix.T, eps, max_rank)
        r_subset = Ctrans.T
        Z_core = Ztrans.T
        rank = r_subset.shape[0]  # r_i-1 = min(r_max, r_delta_i)

        Z_core = tl.reshape(Z_core, [r_g, 2, rank])
        Zj_list.append(Z_core)

        if passed_core_number == dim-2:
            last_core = tl.reshape(r_subset, [rank, 2, 1])
            Zj_list.append(last_core)
        
        end_decomp = tm.time()
        elapsed_time_decomp += (end_decomp - start_decomp)
        
        r_g = rank
        TTRank_new.append(rank)
        TTRank_f1[passed_core_number+1] = rank
        TTRank_f2[passed_core_number+1] = rank
        print(f"Tensor contraction {TTint_contract_f1f2.shape} -> Matrix {skTT_matrix.shape} -> rank revealing -> Z core {Z_core.shape}")

        # Mapping between r/c selection and tensor index pivots
        start_basis = tm.time()
        
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

        # Update TTRank
        TTRank_f1[passed_core_number] = interp_I_f1_gBasis[passed_core_number+1].shape[0]
        TTRank_f2[passed_core_number] = interp_I_f2_gBasis[passed_core_number+1].shape[0]
        passed_core_number += 1
        if sketch_number == 0:
            contract_core_number = dim - passed_core_number
        
        end_basis = tm.time()
        elapsed_time_basis += (end_basis - start_basis)
    
    TTRank_new.append(1)

    end_time_QTTM = tm.time()

    print(f"Runtime of the entire QTTM algorithm: {(end_time_QTTM - start_time_QTTM) * 1000:.2f} ms. The detailed profiling statistics:")
    print(f"Elapsed time of caching sketched cores: {elapsed_time_skcache * 1000:.2f} ms")
    print(f"Elapsed time of sketching QTT: {elapsed_time_sketching * 1000:.2f} ms")
    print(f"Elapsed time of partial contraction: {elapsed_time_contraction * 1000:.2f} ms")
    print(f"Elapsed time of hadamard product: {elapsed_time_product * 1000:.2f} ms")
    print(f"Elapsed time of TTID decomposition: {elapsed_time_decomp * 1000:.2f} ms")
    print(f"Elapsed time of basis update: {elapsed_time_basis * 1000:.2f} ms")
    print(f"Elapsed time of sketching + contraction + decomposition: {(elapsed_time_skcache + elapsed_time_sketching + elapsed_time_contraction + elapsed_time_decomp) * 1000:.2f} ms")

    return interp_I_f1_gBasis, TTRank_new, Zj_list
