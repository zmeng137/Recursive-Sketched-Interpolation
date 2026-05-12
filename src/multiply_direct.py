import time as tm
import numpy as np
import tensorly as tl
from tensorly.tt_tensor import TTTensor
from tensorly.random import random_tt
from tensorly import tt_to_tensor
from tt_rounding import TT_rounding

# Direct O(Chi-4th) approach for Hadamard product of two tensor trains  
def HadamardTT_direct(tt1, tt2):
    assert len(tt1) == len(tt2), "Tensor dimensions of f1 and f2 do not match."
    d = len(tt1)   # Tensor dimension
    factors1 = [tt1[i].copy() for i in range(d)]
    factors2 = [tt2[i].copy() for i in range(d)]
    
    start_t = tm.time()
    tt_product_factors = []
    
    for k in range(d):
        core1 = factors1[k]
        core2 = factors2[k]
        
        # Get dimensions
        r1_prev, n_k, r1_next = core1.shape
        r2_prev, n_k_check, r2_next = core2.shape
        
        if n_k != n_k_check:
            raise ValueError(f"Physical dimensions must match at core {k}")
        
        # The Hadamard product operation for TT cores using copy tensor:
        # Result_k[(α1, α2), i_k, (β1, β2)] = Core1_k[α1, i_k, β1] * Core2_k[α2, i_k, β2]
        
        # Using einsum: multiply cores with matching physical index i
        product_core = np.einsum('aib,cid->acibd', core1, core2)
        
        # Result shape: (r1_prev, r2_prev, n_k, r1_next, r2_next)
        # Reshape to: (r1_prev*r2_prev, n_k, r1_next*r2_next)
        
        new_r_prev = r1_prev * r2_prev
        new_r_next = r1_next * r2_next
        
        # Transpose to (r1_prev, r2_prev, r1_next, r2_next, n_k)
        product_core = product_core.transpose(0, 1, 3, 4, 2)
        # Reshape to (r1_prev*r2_prev, r1_next*r2_next, n_k)
        product_core = product_core.reshape(new_r_prev, new_r_next, n_k)
        # Transpose back to (r1_prev*r2_prev, n_k, r1_next*r2_next)
        product_core = product_core.transpose(0, 2, 1)
        
        tt_product_factors.append(product_core)
    
    end_t = tm.time()
    print(f"Runtime of the direct product algorithm: {(end_t - start_t) * 1000:.2f} ms.")
    
    return tt_product_factors

# Direct O(Chi-4th) approach for Hadamard product of more than two tensor trains
def HadamardTT_direct_fs(tt_dict):
    """
    Compute Hadamard product of multiple tensor trains.
    
    Parameters:
    -----------
    tt_dict : dict
        Dictionary containing tensor trains with keys like 'tt1', 'tt2', 'tt3', etc.
        Each tensor train is a list of cores.
    
    Returns:
    --------
    list
        Tensor train cores representing the Hadamard product.
    """
    # Extract tensor trains from dictionary
    tt_list = [tt_dict[key] for key in sorted(tt_dict.keys())]
    
    if len(tt_list) == 0:
        raise ValueError("Dictionary must contain at least one tensor train")
    
    # Verify all tensor trains have the same dimension
    d = len(tt_list[0])
    for i, tt in enumerate(tt_list):
        if len(tt) != d:
            raise ValueError(f"Tensor train {i} has dimension {len(tt)}, expected {d}")
    
    # Copy all factors
    all_factors = [[tt[i].copy() for i in range(d)] for tt in tt_list]
    
    start_t = tm.time()
    tt_product_factors = []
    
    for k in range(d):
        # Get all cores at position k
        cores = [factors[k] for factors in all_factors]
        
        # Verify physical dimensions match
        n_k = cores[0].shape[1]
        for i, core in enumerate(cores):
            if core.shape[1] != n_k:
                raise ValueError(f"Physical dimensions must match at core {k}")
        
        # Start with the first core
        product_core = cores[0]
        r_prev_accumulated = cores[0].shape[0]
        r_next_accumulated = cores[0].shape[2]
        
        # Sequentially multiply with remaining cores
        for core in cores[1:]:
            r_prev_new, _, r_next_new = core.shape
            
            # Hadamard product: Result[(α_acc, α_new), i_k, (β_acc, β_new)] 
            #                  = Product[α_acc, i_k, β_acc] * Core[α_new, i_k, β_new]
            
            # Using einsum: multiply cores with matching physical index
            product_core = np.einsum('aib,cid->acibd', product_core, core)
            
            # Current shape: (r_prev_acc, r_prev_new, n_k, r_next_acc, r_next_new)
            # Reshape to: (r_prev_acc*r_prev_new, n_k, r_next_acc*r_next_new)
            
            new_r_prev = r_prev_accumulated * r_prev_new
            new_r_next = r_next_accumulated * r_next_new
            
            # Transpose to (r_prev_acc, r_prev_new, r_next_acc, r_next_new, n_k)
            product_core = product_core.transpose(0, 1, 3, 4, 2)
            # Reshape to (new_r_prev, new_r_next, n_k)
            product_core = product_core.reshape(new_r_prev, new_r_next, n_k)
            # Transpose back to (new_r_prev, n_k, new_r_next)
            product_core = product_core.transpose(0, 2, 1)
            
            # Update accumulated ranks
            r_prev_accumulated = new_r_prev
            r_next_accumulated = new_r_next
        
        tt_product_factors.append(product_core)
    
    end_t = tm.time()
    print(f"Runtime of the direct product algorithm: {(end_t - start_t) * 1000:.2f} ms.")
    print(f"Number of tensor trains multiplied: {len(tt_list)}")
    
    return tt_product_factors