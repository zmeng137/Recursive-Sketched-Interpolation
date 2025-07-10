import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt

def populate_tensor_fromfunction(dims, func):
    # Populate tensor using numpy.fromfunction
    def array_func(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
        return func(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)    
    
    # Use fromfunction to create the tensor
    tensor_data = np.fromfunction(array_func, dims, dtype=int)
    return tl.tensor(tensor_data)

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

def integral_qtt(QTT, integral_dim, order=0):
    tensor_dim = len(QTT)
    assert integral_dim <= tensor_dim, "integral dimension should be smaller than or equal to tensor dimension" 
    
    if order == 0:  # Left to right integral
        # Compute the first integral
        first_core = QTT[0]
        integral = 0.5 * first_core[:,0,:] + 0.5 * first_core[:,1,:]
        
        # Compute the following integral
        for i in range(1, integral_dim):
            core = QTT[i]
            sub_int = 0.5 * core[:,0,:] + 0.5 * core[:,1,:]
            integral = integral @ sub_int
        return integral        
    else:
        # Compute the last integral
        last_core = QTT[-1]
        integral = 0.5 * last_core[:,0,:] + 0.5 * last_core[:,1,:]

        # Compute the following integral
        for i in range(tensor_dim - 2, tensor_dim - integral_dim - 1, -1):
            core = QTT[i]
            sub_int = 0.5 * core[:,0,:] + 0.5 * core[:,1,:]
            integral = sub_int @ integral
        return integral



