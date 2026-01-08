import os
import sys
import time as tm
import numpy as np
import math as ma
import tensorly as tl

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MLA-Toolkit', 'py'))
from interpolation import interpolative_prrldu
from utils import adj_ttcore_contract
from sketch import tt_sketching_cache

# Compute the g(f(x)) where f(x) is a TT-approximated function
def NonlinearMapTT_RSI(TT_f1, g_func, contract_core_number, max_rank, eps, sketch_dim, seed):
    dim = len(TT_f1)     # Tensor dimension
    TT_f1_physical_dims = [TT_f1[i].shape[1] for i in range(dim)]  # Physical mode size

    # Copy to prevent in-place modification of input TT
    TT_cores_f1 = [TT_f1[i].copy() for i in range(dim)]
    TT_physical_dims = TT_f1_physical_dims.copy()
    
    # Preparation: Before start ...
    start_time = tm.time()
    passed_core_number = 0   # Iteration number
    r_g = 1                  # Next TT-Rank of g 
    TT_Cores_g = []          # Interpolation Zj cores
    TTRank_g = [1]           # TT-rank list of g
    interp_I_gBasis = {}     # Initialize g's interpolation index set
    interp_I_gBasis[0] = []
    
    # Elapsed time
    elapsed_time_sketching = 0
    elapsed_time_contraction = 0
    elapsed_time_decomp = 0
    elapsed_time_product = 0
    elapsed_time_updpivot = 0
    elapsed_time_reinterp = 0

    # Randomized sketching cache
    start_time_sketch_cache = tm.time()
    sk_tail_number = dim - contract_core_number
    f1_sk_tt = tt_sketching_cache(TT_cores_f1, sk_tail_number, sketch_dim, seed)
    end_time_sketch_cache = tm.time()
    elapsed_time_skcache = end_time_sketch_cache - start_time_sketch_cache

    # Recursive Interpolative Sketching 
    while passed_core_number < dim-1:
        # Check if we need to sketch the last cores
        physical_dim = TT_physical_dims[passed_core_number]
        free_phydim_number = dim - passed_core_number - contract_core_number
        residual_size = ma.prod(TT_physical_dims[-free_phydim_number:])
        
        # If we need to sketch or not in this iter
        sketch_number = 0
        if free_phydim_number > 0:
            if sketch_dim < residual_size:
                sketch_number = free_phydim_number  
            else:
                contract_core_number = dim - passed_core_number        
        print(f"# sketch: {sketch_number}, # contraction {contract_core_number}, # passed core {passed_core_number}.")

        ''' (i) Sketching and Contraction '''
        # Partial contraction of f's TT-cores
        start_contraction = tm.time()
        f1_contract = TT_cores_f1[passed_core_number]
        for i in range(1, contract_core_number):
            f1_contract = np.tensordot(f1_contract, TT_cores_f1[passed_core_number + i], axes=([len(f1_contract.shape)-1],[0]))
        end_contraction = tm.time()
        elapsed_time_contraction += (end_contraction - start_contraction)

        # Randomized sketching of f's QTT
        start_sketching = tm.time()
        if sketch_number > 0:
            skCore_f1 = np.zeros([f1_contract.shape[-1], sketch_dim])
            for l in range(sketch_dim):
                sk_f1 = f1_sk_tt[passed_core_number][:,l,:].copy()
                for i in range(passed_core_number + 1, sk_tail_number):
                    sk_f1 = sk_f1 @ f1_sk_tt[i][:,l,:]
                skCore_f1[:, l] = np.squeeze(sk_f1, axis=-1)
            f1_contract = f1_contract @ skCore_f1

        end_sketching = tm.time()
        elapsed_time_sketching += (end_sketching - start_sketching)

        ''' (ii) Hadamard product and new TT-core decomposition '''
        # Hadamard product
        start_product = tm.time()
        TTint_contract_f1map = g_func(f1_contract)
        end_product = tm.time()
        elapsed_time_product += (end_product - start_product)

        # New pivot selection (decomposition)
        start_decomp = tm.time()
        shape_contract_t = TTint_contract_f1map.shape 
        mat_row = ma.prod(shape_contract_t[0:2])
        mat_col = ma.prod(shape_contract_t[2:])
        skTT_matrix = tl.reshape(TTint_contract_f1map, [mat_row, mat_col])    
        
        # One-side ID: Z Interpolation core
        Ctrans, Ztrans, pr, _ = interpolative_prrldu(skTT_matrix.T, eps, max_rank)
        r_subset = Ctrans.T
        Z_core = Ztrans.T
        rank = r_subset.shape[0]  # r_i-1 = min(r_max, r_delta_i)

        Z_core = tl.reshape(Z_core, [r_g, physical_dim, rank])
        TT_Cores_g.append(Z_core)
        if passed_core_number == dim - 2:
            last_core = tl.reshape(r_subset, [rank, TT_physical_dims[-1], 1])
            TT_Cores_g.append(last_core)
        
        end_decomp = tm.time()
        elapsed_time_decomp += (end_decomp - start_decomp)
        
        r_g = rank
        TTRank_g.append(rank)
        print(f"Tensor contraction {TTint_contract_f1map.shape} -> Matrix {skTT_matrix.shape} -> rank revealing -> Z core {Z_core.shape}")

        # Mapping between r/c selection and tensor index pivots
        start_updbasis = tm.time()
        if passed_core_number == 0:
            interp_I_gBasis[passed_core_number+1] = np.array(pr).reshape(-1, 1)
        else:
            I = np.empty([rank, passed_core_number+1])
            prev_I = interp_I_gBasis[passed_core_number]
            for j in range(rank):
                p_I_idx = pr[j] // physical_dim
                c_i_idx = pr[j] % physical_dim
                I[j,0:passed_core_number] = prev_I[p_I_idx]
                I[j,passed_core_number] = c_i_idx
            interp_I_gBasis[passed_core_number+1] = I
        end_updbasis = tm.time()
        elapsed_time_updpivot += (end_updbasis - start_updbasis)

        ''' (iii) Re-interpolation of f1 and f2 TT-core at the current iteration after sketching + product + decomposition '''
        start_reinterp = tm.time()
        if passed_core_number < dim -2:
            # Cache contraction to approximate values not included in interpolation
            I_gbasis_next = interp_I_gBasis[passed_core_number + 1]
            I_gbasis_curr = interp_I_gBasis[passed_core_number]
            cache_contract_f1 = adj_ttcore_contract(TT_cores_f1[passed_core_number], TT_cores_f1[passed_core_number + 1])
            
            new_core_shape_f1 = [len(I_gbasis_next), TT_cores_f1[passed_core_number+1].shape[1], TT_cores_f1[passed_core_number+1].shape[2]]    
            curr_core_f1 = np.empty(new_core_shape_f1)

            # Get approximated TT-cores interpolated by new g basis
            for i in range(len(I_gbasis_next)):
                curr_pivot_i = I_gbasis_next[i]
                last_pivot = curr_pivot_i[-1].astype(int)    
                arg_idx = 0
                if passed_core_number > 0:
                    prev_pivot = curr_pivot_i[0:-1]
                    matches = np.all(I_gbasis_curr == prev_pivot, axis = 1)
                    arg_idx = np.where(matches)[0][0]
                curr_core_f1[i, :, :] = cache_contract_f1[arg_idx, last_pivot, :, :]
            
            TT_cores_f1[passed_core_number + 1] = curr_core_f1

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
    print(f"Elapsed time of nonlinear mapping: {elapsed_time_product * 1000:.2f} ms")
    print(f"Elapsed time of interpolative decomposition: {elapsed_time_decomp * 1000:.2f} ms")
    print(f"Elapsed time of basis update: {elapsed_time_updpivot * 1000:.2f} ms")

    return TT_Cores_g, TTRank_g, interp_I_gBasis 