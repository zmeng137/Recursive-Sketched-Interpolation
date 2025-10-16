import os
import sys
import time as tm
import numpy as np
import math as ma
import tensorly as tl

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MLA-Toolkit', 'py'))
from interpolation import interpolative_prrldu
from qtt import adj_ttcore_contract, qtt_sketching_cache

# Compute g=F(f(x)) where f(x) is a QTT-approximated function
def functional_qtt(func,
            TT_cores_f, interp_I_f, TTRank_f,
            contract_core_number, max_rank, eps,
            randomFlag, seed, skLayer):
    # Preparation for g's I interpolation basis
    dim = len(TTRank_f) - 1  # Tensor dimension
    TTRank_f = TTRank_f.copy()
    interp_I_f_gBasis = interp_I_f.copy()
    start_time = tm.time()
    
    # Randomized sketching cache
    start_time_sketch_cache = tm.time()
    f_sk_tt, skLayer = qtt_sketching_cache(TT_cores_f, randomFlag, seed, skLayer)

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

        start_contraction = tm.time()
        if passed_core_number > 0:
            # Cache contraction to approximate values not included in interpolation
            I_gbasis_curr = interp_I_f_gBasis[passed_core_number]
            I_gbasis_prev = interp_I_f_gBasis[passed_core_number - 1]
            cache_contract_f = adj_ttcore_contract(TT_cores_f[passed_core_number-1], TT_cores_f[passed_core_number])

            # Re-interpolate to get the current core   
            curr_core_f = np.empty([len(I_gbasis_curr), 2, len(interp_I_f_gBasis[passed_core_number + 1])])
          
            # Get approximated TT-cores interpolated by new g basis
            for i in range(len(I_gbasis_curr)):
                curr_pivot_i = I_gbasis_curr[i]
                last_pivot = curr_pivot_i[-1].astype(int)    
                arg_idx = 0
                if passed_core_number > 1:
                    prev_pivot = curr_pivot_i[0:-1]
                    matches = np.all(I_gbasis_prev == prev_pivot, axis = 1)
                    arg_idx = np.where(matches)[0][0]
                curr_core_f[i, :, :] = cache_contract_f[arg_idx, last_pivot, :, :]
             
            TT_cores_f[passed_core_number] = curr_core_f
             
        # Partial contraction of f 
        f_contract = TT_cores_f[passed_core_number]
        for i in range(1, contract_core_number):
            f_contract = np.tensordot(f_contract, TT_cores_f[passed_core_number + i], axes=([len(f_contract.shape)-1],[0]))
        end_contraction = tm.time()
        elapsed_time_contraction += (end_contraction - start_contraction)

        # Randomized sketching of f's QTT
        start_sketching = tm.time()
        if sketch_number > 0:
            skCore_f = np.zeros([f_contract.shape[-1], skLayer])
            for l in range(skLayer):
                sk_f = f_sk_tt[l][-1]
                for i in range(dim - 2, dim - sketch_number - 1, -1):
                    sub_int_f = f_sk_tt[l][i]
                    sk_f = sub_int_f @ sk_f
                skCore_f[:, l] = np.squeeze(sk_f, axis=-1)
            f_contract = f_contract @ skCore_f
        end_sketching = tm.time()
        elapsed_time_sketching += (end_sketching - start_sketching)

        # Applying the functional F(f(x))
        start_product = tm.time()
        TT_functional = func(f_contract)
        end_product = tm.time()
        elapsed_time_product += (end_product - start_product)

        # New pivot selection (decomposition)
        start_decomp = tm.time()
        
        shape_contract_t = TT_functional.shape 
        mat_row = ma.prod(shape_contract_t[0:2])
        mat_col = ma.prod(shape_contract_t[2:])
        skTT_matrix = tl.reshape(TT_functional, [mat_row, mat_col])    
        
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
        TTRank_f[passed_core_number+1] = rank
        print(f"Tensor contraction {TT_functional.shape} -> Matrix {skTT_matrix.shape} -> rank revealing -> Z core {Z_core.shape}")

        # Mapping between r/c selection and tensor index pivots
        start_basis = tm.time()
        
        if passed_core_number == 0:
            interp_I_f_gBasis[passed_core_number+1] = np.array(pr).reshape(-1, 1)
        else:
            I = np.empty([rank, passed_core_number+1])
            prev_I = interp_I_f_gBasis[passed_core_number]
            for j in range(rank):
                p_I_idx = pr[j] // 2
                c_i_idx = pr[j] % 2
                I[j,0:passed_core_number] = prev_I[p_I_idx]
                I[j,passed_core_number] = c_i_idx
            interp_I_f_gBasis[passed_core_number+1] = I

        # Update TTRank
        TTRank_f[passed_core_number] = interp_I_f_gBasis[passed_core_number+1].shape[0]
        passed_core_number += 1
        if sketch_number == 0:
            contract_core_number = dim - passed_core_number
        
        end_basis = tm.time()
        elapsed_time_basis += (end_basis - start_basis)
    
    TTRank_new.append(1)

    end_time = tm.time()

    print(f"Runtime of the entire algorithm: {(end_time - start_time) * 1000:.2f} ms. The detailed profiling statistics:")
    print(f"Elapsed time of caching sketched cores: {elapsed_time_skcache * 1000:.2f} ms")
    print(f"Elapsed time of sketching QTT: {elapsed_time_sketching * 1000:.2f} ms")
    print(f"Elapsed time of partial contraction: {elapsed_time_contraction * 1000:.2f} ms")
    print(f"Elapsed time of functional application: {elapsed_time_product * 1000:.2f} ms")
    print(f"Elapsed time of TTID decomposition: {elapsed_time_decomp * 1000:.2f} ms")
    print(f"Elapsed time of basis update: {elapsed_time_basis * 1000:.2f} ms")
    print(f"Elapsed time of sketching + contraction + decomposition: {(elapsed_time_skcache + elapsed_time_sketching + elapsed_time_contraction + elapsed_time_decomp) * 1000:.2f} ms")

    return interp_I_f_gBasis, TTRank_new, Zj_list
