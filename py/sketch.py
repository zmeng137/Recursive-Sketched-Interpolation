import numpy as np
import random as rd

# Generate a random array with entries from standard normal distribution.
def random_normal_tensor(shape, complex=False):
    if complex:
        # Complex standard normal: real and imaginary parts are N(0, 1/2)
        # So that E[|z|²] = 1
        real_part = np.random.randn(*shape if isinstance(shape, tuple) else (shape,)) / np.sqrt(2)
        imag_part = np.random.randn(*shape if isinstance(shape, tuple) else (shape,)) / np.sqrt(2)
        return real_part + 1j * imag_part
    else:
        # Real standard normal: N(0, 1)
        return np.random.randn(*shape if isinstance(shape, tuple) else (shape,))

def tt_sketching_cache(tt_cores, sketch_tail_no, over_sampling, seed):
    dim = len(tt_cores)
    shape = [tt_cores[i].shape[1] for i in range(dim)]   
    print(f"Performing TT random sketching cache... the order-{dim} tensor has shape of {shape}")

    # Formalize the new sketched TT 
    tt_sketched = tt_cores[dim - sketch_tail_no:].copy()
    
    for i in range(sketch_tail_no):
        rd.seed(seed * i + 1)
        core_pos = dim - sketch_tail_no + i
        sk_shape = (shape[core_pos], over_sampling)
        sk_tensor = random_normal_tensor(sk_shape)
        tt_sketched[i] = np.einsum('ijk,jl->ilk', tt_sketched[i], sk_tensor)        

    return tt_sketched