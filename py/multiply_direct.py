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

'''
# Example usage
if __name__ == "__main__":
    print("Example: Hadamard product of two tensor trains")
    print("=" * 50)
    
    # Generate two random tensor trains with the same shape
    shape = [10, 10, 10, 10, 10]
    ttrank_tt1 = [1, 10, 10, 10, 10, 1]
    ttrank_tt2 = [1, 10, 10, 10, 10, 1]

    # Generate random TT tensors
    tt1 = random_tt(shape=shape, rank=ttrank_tt1, full=False)
    tt2 = random_tt(shape=shape, rank=ttrank_tt2, full=False)
    
    print(f"Tensor shape: {shape}")
    print(f"\nTT1 core shapes: {[core.shape for core in tt1.factors]}")
    print(f"TT2 core shapes: {[core.shape for core in tt2.factors]}")
    
    # Compute Hadamard product in TT format
    tt_product = HadamardTT_direct(tt1, tt2)
    tt_product_trunc = TT_rounding(tt_product, 1e-16, 20)
    
    print(f"\nHadamard product TT core shapes: {[core.shape for core in tt_product]}")
    
    # Verify correctness by reconstructing full tensors
    tensor1 = tt_to_tensor(tt1)
    tensor2 = tt_to_tensor(tt2)
    tensor_product = tt_to_tensor(tt_product)
    tensor_product_trunc = tt_to_tensor(tt_product_trunc)
    
    # Expected result: element-wise product
    expected_product = tensor1 * tensor2
    
    # Calculate error
    error = np.linalg.norm(tensor_product - expected_product) / np.linalg.norm(expected_product)
    error_trunc = np.linalg.norm(tensor_product_trunc - expected_product) / np.linalg.norm(expected_product)
    
    print(f"\nRelative error: {error:.2e}")
    print(f"Compressed relative error: {error_trunc:.2e}")
    print(f"Success: {error < 1e-10}")
    
    # Show rank growth
    r1_ranks = [tt1.factors[k].shape[2] for k in range(len(tt1.factors)-1)]
    r2_ranks = [tt2.factors[k].shape[2] for k in range(len(tt2.factors)-1)]
    product_ranks = [tt_product[k].shape[2] for k in range(len(tt_product)-1)]
    compressed_ranks = [tt_product_trunc[k].shape[2] for k in range(len(tt_product_trunc)-1)]

    print(f"\nRank growth (internal ranks):")
    print(f"  TT1 ranks: {r1_ranks}")
    print(f"  TT2 ranks: {r2_ranks}")
    print(f"  Product ranks: {product_ranks}")
    print(f"  Compressed Product ranks: {compressed_ranks}")
    print(f"  Expected (r1*r2): {[r1_ranks[i]*r2_ranks[i] for i in range(len(r1_ranks))]}")
'''