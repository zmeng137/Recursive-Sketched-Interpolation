import time as tm
import numpy as np
import tensorly as tl
from tensorly.tt_tensor import TTTensor
from tensorly.random import random_tt
from tensorly import tt_to_tensor


def tt_hadamard_product(tt1, tt2):
    start_t = tm.time()

    # Extract factors if TTTensor objects
    if isinstance(tt1, TTTensor):
        factors1 = tt1.factors
    else:
        factors1 = tt1
        
    if isinstance(tt2, TTTensor):
        factors2 = tt2.factors
    else:
        factors2 = tt2
    
    if len(factors1) != len(factors2):
        raise ValueError("Both tensor trains must have the same number of cores")
    
    d = len(factors1)  # number of cores
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
        # 
        # For element-wise product, we need:
        # (TT1 ⊙ TT2)[i1, i2, ..., id] = TT1[i1, i2, ..., id] * TT2[i1, i2, ..., id]
        #
        # The copy tensor approach: Result_k[(α1, α2), i_k, (β1, β2)] 
        #                                    = Core1_k[α1, i_k, β1] * Core2_k[α2, i_k, β2]
        #
        # core1: (r1_prev, n_k, r1_next)
        # core2: (r2_prev, n_k, r2_next)
        
        # Using einsum: multiply cores with matching physical index i
        product_core = np.einsum('aib,cid->acibd', core1, core2)
        
        # Result shape: (r1_prev, r2_prev, n_k, r1_next, r2_next)
        # Need to reshape to: (r1_prev*r2_prev, n_k, r1_next*r2_next)
        # The multi-index (α1, α2) should be in C-order: α2 + r2_prev * α1
        
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
        
    print(f"Runtime of the entire algorithm: {(end_t - start_t) * 1000:.2f} ms.")

    return TTTensor(tt_product_factors)


# Example usage
if __name__ == "__main__":
    print("Example: Hadamard product of two tensor trains")
    print("=" * 50)
    
    # Generate two random tensor trains with the same shape
    shape = [7, 9, 4, 5, 6, 10, 5, 3, 11, 5]
    rank1 = [1, 4, 15, 101, 80, 30, 110, 25, 18, 3, 1]
    rank2 = [1, 3, 29, 100, 107, 230, 19, 301, 29, 2, 1]
    
    # Generate random TT tensors
    tt1 = random_tt(shape=shape, rank=rank1, full=False)
    tt2 = random_tt(shape=shape, rank=rank2, full=False)
    
    print(f"Tensor shape: {shape}")
    print(f"\nTT1 core shapes: {[core.shape for core in tt1.factors]}")
    print(f"TT2 core shapes: {[core.shape for core in tt2.factors]}")
    
    # Compute Hadamard product in TT format
    tt_product = tt_hadamard_product(tt1, tt2)
    
    print(f"\nHadamard product TT core shapes: {[core.shape for core in tt_product.factors]}")
    
    # Verify correctness by reconstructing full tensors
    tensor1 = tt_to_tensor(tt1)
    tensor2 = tt_to_tensor(tt2)
    tensor_product = tt_to_tensor(tt_product)
    
    # Expected result: element-wise product
    expected_product = tensor1 * tensor2
    
    # Calculate error
    error = np.linalg.norm(tensor_product - expected_product) / np.linalg.norm(expected_product)
    
    print(f"\nRelative error: {error:.2e}")
    print(f"Success: {error < 1e-10}")
    
    # Show rank growth
    r1_ranks = [tt1.factors[k].shape[2] for k in range(len(tt1.factors)-1)]
    r2_ranks = [tt2.factors[k].shape[2] for k in range(len(tt2.factors)-1)]
    product_ranks = [tt_product.factors[k].shape[2] for k in range(len(tt_product.factors)-1)]
    
    print(f"\nRank growth (internal ranks):")
    print(f"  TT1 ranks: {r1_ranks}")
    print(f"  TT2 ranks: {r2_ranks}")
    print(f"  Product ranks: {product_ranks}")
    print(f"  Expected (r1*r2): {[r1_ranks[i]*r2_ranks[i] for i in range(len(r1_ranks))]}")