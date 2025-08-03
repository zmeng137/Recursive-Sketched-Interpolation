import numpy as np
import itertools
import tensorly as tl
import matplotlib.pyplot as plt

# Populate tensor using numpy.fromfunction
def populate_tensor_fromfunction(dims, func):
    # Populate tensor using numpy.fromfunction
    def array_func(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12):
        return func(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)    
    
    # Use fromfunction to create the tensor
    tensor_data = np.fromfunction(array_func, dims, dtype=int)
    return tl.tensor(tensor_data)

# Scatter plot for f1, f2, g
def scatter_plot_f1f2(x_tensor, g_tensor, f1_tensor = None, f2_tensor = None):
    plt.figure()
    plt.scatter(x_tensor, g_tensor, s=8, alpha=0.8, linewidth=0.5, label='g')
    if f1_tensor is not None:
        plt.scatter(x_tensor, f1_tensor, s=4, alpha=0.8, linewidth=0.5, label='f1')
    if f2_tensor is not None:
        plt.scatter(x_tensor, f2_tensor, s=4, alpha=0.8, linewidth=0.5, label='f2')
    plt.legend()
    plt.grid()
    plt.savefig("f1_f2_g.png")
    return

# Generate quantics representation of a continuous function
def quantics_generation(func, digit):
    shape = tuple([2] * digit)   # Tensor shape 2^n
    x_tensor = np.zeros(shape)   # Quantics x tensor
    f_tensor = np.zeros(shape)   # Quantics function 
    
    # Generate all possible combinations of indices (0,1) for n dimensions
    for indices in itertools.product([0, 1], repeat = digit):
        # Calculate the value using the formula: x1/2 + x2/2^2 + ... + xn/2^n
        value = sum(x / (2 ** (i + 1)) for i, x in enumerate(indices))
        x_tensor[indices] = value
        f_tensor[indices] = func(value)
    
    return x_tensor, f_tensor

# Contract two adjacent TT-cores
def adj_ttcore_contract(core1, core2):
    if core1.shape[2] != core2.shape[0]:
        raise ValueError(f"Incompatible shapes: core1 rank {core1.shape[2]} != core2 rank {core2.shape[0]}")
    
    # Perform the contraction via einsum
    contracted = np.einsum('air,rjb->aijb', core1, core2)
    
    return contracted

# Union rows of two arrays with a maximum row limit
def union_rows_bounded(A, B, max_rows):
    A = np.array(A)
    B = np.array(B)
    
    # Find rows in B that are not in A
    mask = ~np.any(np.all(A[:, None] == B, axis=2), axis=0)
    new_rows = B[mask]
    
    # Calculate how many rows we can add
    current_rows = len(A)
    max_new_rows = max_rows - current_rows
    
    if max_new_rows <= 0:
        return A  # Already at or above limit
    
    # Take only the first max_new_rows rows (by row order)
    rows_to_add = new_rows[:max_new_rows]
    
    # Concatenate
    result = np.vstack([A, rows_to_add]) if len(rows_to_add) > 0 else A

    return result

# Union rows of two arrays with a maximum row limit, randomly sampling if necessary
def union_rows_bounded_random(A, B, max_rows):
    # Get unique rows from both arrays
    C = np.unique(np.vstack([A, B]), axis=0)
    
    # If we have more rows than the limit, randomly sample
    if len(C) > max_rows:
        # Randomly select indices without replacement
        selected_indices = np.random.choice(len(C), size=max_rows, replace=False)
        C = C[selected_indices]
    
    return C

# Integrate the QTT (right to left) in a given digit number
def integral_qtt(QTT, integral_dim):
    tensor_dim = len(QTT)
    assert integral_dim <= tensor_dim, "integral dimension should be smaller than or equal to tensor dimension" 
    
    # Compute the last integral
    last_core = QTT[-1]
    integral = 0.5 * last_core[:,0,:] + 0.5 * last_core[:,1,:]

    # Compute the following integral
    for i in range(tensor_dim - 2, tensor_dim - integral_dim - 1, -1):
        core = QTT[i]
        sub_int = 0.5 * core[:,0,:] + 0.5 * core[:,1,:]
        integral = sub_int @ integral
    
    # Newly-integrated QTT
    QTT_new = QTT[0: tensor_dim - integral_dim].copy()
    QTT_new[-1] = QTT_new[-1] @ integral.reshape(-1, 1)

    return QTT_new

# Integrate the QTT: Only contract the integral tensor with every core, but no TT contraction
def Qintegral_TT(QTT):
    dim = len(QTT)   # Number of TT-cores
    TT_int = []      # Integral TT
    for i in range(dim):
        core = QTT[i]
        int_core = 0.5 * core[:,0,:] + 0.5 * core[:,1,:]
        TT_int.append(int_core) 
    return TT_int

# Query a value of a function from its QTT at a specific position    
def value_query_QTT(QTT, TTRank, pos):
    dim = len(QTT)
    interm_core = QTT[0][0,pos[0],:]  # Initial TT-core as the first intermediate core
    
    # Fix-index contraction 
    for p in range(dim-1):
        free = pos[p+1]
        left_bond = TTRank[p+1]
        right_bond = TTRank[p+2]
        merge = np.zeros(right_bond)
        for i in range(right_bond):
            for j in range(left_bond):
                merge[i] += interm_core[j] * QTT[p+1][j, free, i] 
        interm_core = merge
    return interm_core[0]