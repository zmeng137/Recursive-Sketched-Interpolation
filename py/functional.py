import os
import sys
import time as tm
import numpy as np
import math as ma
import tensorly as tl

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MLA-Toolkit', 'py'))
from interpolation import interpolative_prrldu
from qtt import adj_ttcore_contract, qtt_sketching_cache

# Compute g(f(x)) where f(x) is a QTT-approximated function
def functional_qtt(g_func, TTcores_f, interp_I_f, 
                   contract_core_number, max_rank, eps,
                   randomFlag, seed, skLayer):
    # Randomized sketching cache
    start_time = tm.time()
    start_time_sketch_cache = tm.time()
    f_sk_tt, skLayer = qtt_sketching_cache(TTcores_f, randomFlag, seed, skLayer)
    end_time_sketch_cache = tm.time()
    elapsed_time_skcache = end_time_sketch_cache - start_time_sketch_cache
    
    # Preliminary: Before iteration...
    dim = len(TTcores_f)    # Tensor dimension
    passed_core_number = 0  # Iteration number
    r_g = 1                 # Next TT-Rank of g 
    TTcores_g = []          # Interpolation Zj cores
    TTRank_g = [1]          # TT-rank list of g
    interp_I_gBasis = {}    # Initialize g's interpolation index set
    interp_I_gBasis[0] = []
    interp_I_gBasis[1] = interp_I_f[1].copy()
    
    # Elapsed time
    elapsed_time_sketching = 0
    elapsed_time_contraction = 0
    elapsed_time_decomp = 0
    elapsed_time_functional = 0
    elapsed_time_updpivot = 0
    elapsed_time_reinterp = 0
    
    # Recursive Interpolative Sketching 
    while passed_core_number < dim-1:
        # Check if we need to integrate the last cores
        temp = dim - passed_core_number - contract_core_number
        sketch_number = temp if temp > 0 else 0 
        free_dim = TTcores_f[passed_core_number].shape[1]
        print(f"# sketch: {sketch_number}, # contraction {contract_core_number}, # passed core {passed_core_number}.")
             
        ''' (i) Sketching and Contraction '''
        # Contraction (partially) of f's TT-cores 
        start_contraction = tm.time()
        f_contract = TTcores_f[passed_core_number]
        for i in range(1, contract_core_number):
            f_contract = np.tensordot(f_contract, TTcores_f[passed_core_number + i], axes=([len(f_contract.shape)-1],[0]))
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

        ''' (ii) Functional application and new TT-core decomposition '''
        # Applying the functional g(f(x))
        start_functional = tm.time()
        TT_functional = g_func(f_contract)
        end_functional = tm.time()
        elapsed_time_functional += (end_functional - start_functional)

        # Get the reshaped matrix for interpolative decomposition (New pivot selection)
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

        Z_core = tl.reshape(Z_core, [r_g, free_dim, rank])
        TTcores_g.append(Z_core)

        if passed_core_number == dim - 2:
            last_core = tl.reshape(r_subset, [rank, TTcores_f[-1].shape[1], 1])
            TTcores_g.append(last_core)
        
        end_decomp = tm.time()
        elapsed_time_decomp += (end_decomp - start_decomp)
        
        r_g = rank
        TTRank_g.append(rank)
        print(f"Tensor contraction {TT_functional.shape} -> Matrix {skTT_matrix.shape} -> rank revealing -> Z core {Z_core.shape}")

        # Mapping between r/c selection and tensor index pivots
        start_updbasis = tm.time()
        if passed_core_number == 0:
            interp_I_gBasis[passed_core_number+1] = np.array(pr).reshape(-1, 1)
        else:
            I = np.empty([rank, passed_core_number+1])
            prev_I = interp_I_gBasis[passed_core_number]
            for j in range(rank):
                p_I_idx = pr[j] // free_dim
                c_i_idx = pr[j] % free_dim
                I[j,0:passed_core_number] = prev_I[p_I_idx]
                I[j,passed_core_number] = c_i_idx
            interp_I_gBasis[passed_core_number+1] = I
        
        end_updbasis = tm.time()
        elapsed_time_updpivot += (end_updbasis - start_updbasis)

        ''' (iii) Re-interpolation of f's TT-core at the current iteration after sketching + functional + decomposition '''
        start_reinterp = tm.time()
        if passed_core_number <  dim - 2:
            # Cache contraction to approximate values not included in interpolation
            I_gbasis_next = interp_I_gBasis[passed_core_number + 1]
            I_gbasis_curr = interp_I_gBasis[passed_core_number]
            cache_contract_f = adj_ttcore_contract(TTcores_f[passed_core_number], TTcores_f[passed_core_number + 1])

            # New shape of f's current core after re-interpolation
            new_core_shape = [len(I_gbasis_next), TTcores_f[passed_core_number+1].shape[1], len(interp_I_f[passed_core_number + 2])]   
            curr_core_f = np.empty(new_core_shape)

            # Get approximated TT-cores interpolated by new g basis
            for i in range(len(I_gbasis_next)):
                curr_pivot_i = I_gbasis_next[i]
                last_pivot = curr_pivot_i[-1].astype(int)    
                arg_idx = 0     
                if passed_core_number > 0:
                    prev_pivot = curr_pivot_i[0:-1]
                    matches = np.all(I_gbasis_curr == prev_pivot, axis = 1)
                    arg_idx = np.where(matches)[0][0]
                curr_core_f[i, :, :] = cache_contract_f[arg_idx, last_pivot, :, :]
            
            # New f's TT-core after re-interpolation
            TTcores_f[passed_core_number + 1] = curr_core_f
        
        end_reinterp = tm.time()
        elapsed_time_reinterp += (end_reinterp - start_reinterp)

        passed_core_number += 1
        if sketch_number == 0:
            contract_core_number = dim - passed_core_number
    
    TTRank_g.append(1)
    end_time = tm.time()

    print(f"Runtime of the entire algorithm: {(end_time - start_time) * 1000:.2f} ms. The detailed profiling statistics:")
    print(f"Elapsed time of caching sketched cores: {elapsed_time_skcache * 1000:.2f} ms")
    print(f"Elapsed time of TT sketching: {elapsed_time_sketching * 1000:.2f} ms")
    print(f"Elapsed time of partial contraction: {elapsed_time_contraction * 1000:.2f} ms")
    print(f"Elapsed time of functional application: {elapsed_time_functional * 1000:.2f} ms")
    print(f"Elapsed time of interpolative decomposition: {elapsed_time_decomp * 1000:.2f} ms")
    print(f"Elapsed time of basis update: {elapsed_time_updpivot * 1000:.2f} ms")

    return interp_I_gBasis, TTRank_g, TTcores_g
