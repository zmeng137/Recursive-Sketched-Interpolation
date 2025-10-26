import os
import sys
import numpy as np
import random as rd
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MLA-Toolkit', 'py'))
from tci import slice_first_modes, slice_last_modes

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

# Random/Integral sketching of the last digits of a given quantics tensor train for feature extraction
def qtt_sketching(QTT, sketch_dim, randomFlag, seed, skLayer):
    # Sketching dimension, Random seeds, Sketching layers
    tensor_dim = len(QTT)   
    assert sketch_dim <= tensor_dim, "Sketching dimension should be smaller than or equal to tensor dimension"

    # Formalize the new sketched QTT 
    QTT_new = QTT[0: tensor_dim - sketch_dim].copy()    
    
    # Initialize the last sketched core
    if randomFlag == False:
        skLayer = 1
    new_skcore = np.zeros([QTT_new[-1].shape[0], 2, skLayer])

    # Random sketching
    rd.seed(seed)
    for l in range(skLayer):        
        # Random 2-entry vector
        if randomFlag == True:
            x = rd.random()
            y = 1 - x
        else:
            x = 0.5
            y = 0.5
        print(f"Sketching layer {l}: random vector {(x, y)}")
        
        # Sketching 
        last_core = QTT[-1].copy()  # The last TT-core
        skIntegral = x * last_core[:,0,:] + y * last_core[:,1,:]
        for i in range(tensor_dim - 2, tensor_dim - sketch_dim - 1, -1):
            core = QTT[i].copy()
            sub_int = x * core[:,0,:] + y * core[:,1,:]
            skIntegral = sub_int @ skIntegral

        # Newly-integrated QTT
        result = QTT_new[-1].copy() @ skIntegral.reshape(-1, 1)
        new_skcore[:, :, l] = np.squeeze(result, axis=-1)

    # All sketching appears in the last digit 
    QTT_new[-1] = new_skcore    
    
    return QTT_new

# Random/Integral sketching of the last several digits of a given quantics tensor train for feature extraction
# The sketching TT-cores are kept in list as cache waiting for query
def qtt_sketching_cache(qtt, randomFlag, seed, skLayer):
    dim = len(qtt)   
    
    # Formalize the new sketched QTT 
    qtt_sketched = []

    # No randomization -> integral sketching
    if randomFlag == False:
        skLayer = 1

    # Random or Integral sketching
    rd.seed(seed)
    for l in range(skLayer):        
        if randomFlag == True:
            # Random 2-entry vector
            x = np.random.rand(dim)
            y = 1 - x
        else:
            # Integral 2-entry vector
            x = 0.5 * np.ones(dim)
            y = 0.5 * np.ones(dim)
        print(f"Sketching layer {l}: random vectors {(x[0], y[0])}, ...")
        
        # Sketching
        skTT_1l = []
        for d in range(dim):
            core = qtt[d].copy()
            sketch = x[d] * core[:,0,:] + y[d] * core[:,1,:]
            skTT_1l.append(sketch)

        qtt_sketched.append(skTT_1l)
        skLayer = len(qtt_sketched)
    
    return qtt_sketched, skLayer

# Integrate the QTT (right to left) in a given digit number
def integral_qtt(QTT, integral_dim):
    tensor_dim = len(QTT)
    assert integral_dim <= tensor_dim, "integral dimension should be smaller than or equal to tensor dimension" 
    
    full_int = False
    if integral_dim == tensor_dim:
        integral_dim = tensor_dim - 1
        full_int = True

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

    if full_int == True:
        fc = np.squeeze(QTT_new[0])
        QTT_new = 0.5 * fc[0] + 0.5 * fc[1]        

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

# Plot the pivots used to interpolate the function
def plot_interp_pivots(interp_I, interp_J, x_tensor, y_tensor):
    # Dimension and tensor flattening
    dim = len(y_tensor.shape)
    x_flat = x_tensor.flatten()
    y_flat = y_tensor.flatten()

    # Plot the quantics tensor
    plt.figure()
    plt.plot(x_flat, y_flat)

    # Plot the pivots in every mode. Assmebly of TT-Cores via interpolation sets
    for d in range(dim):
        # Construct TT-cores
        if d == 0:
            right_rank = len(interp_J[2])
            x_piv_val = np.empty([1, 2, right_rank])
            y_piv_val = np.empty([1, 2, right_rank])
            for j in range(right_rank):
                J_slice = interp_J[2][j].astype(int).tolist()
                x_piv_val[0,:,j] = slice_last_modes(x_tensor, J_slice)
                y_piv_val[0,:,j] = slice_last_modes(y_tensor, J_slice)
            
        elif d == dim-1:
            left_rank = len(interp_I[d])
            x_piv_val = np.empty([left_rank, 2, 1])
            y_piv_val = np.empty([left_rank, 2, 1])
            for i in range(left_rank):
                I_slice = interp_I[d][i].astype(int).tolist()
                x_piv_val[i,:,0] = slice_first_modes(x_tensor, I_slice)
                y_piv_val[i,:,0] = slice_first_modes(y_tensor, I_slice)

        else:
            left_rank = len(interp_I[d])
            right_rank = len(interp_J[d+2])
            x_piv_val = np.empty([left_rank, 2, right_rank])
            y_piv_val = np.empty([left_rank, 2, right_rank])
            for i in range(left_rank):
                I_slice = interp_I[d][i].astype(int).tolist()
                for j in range(right_rank):
                    J_slice = interp_J[d+2][j].astype(int).tolist()
                    x_temp = slice_first_modes(x_tensor, I_slice)
                    y_temp = slice_first_modes(y_tensor, I_slice)
                    x_piv_val[i,:,j] = slice_last_modes(x_temp, J_slice)
                    y_piv_val[i,:,j] = slice_last_modes(y_temp, J_slice)

        plt.scatter(x_piv_val, y_piv_val, label=f'{d}-th core pivots')    

    plt.xlabel('x values')
    plt.ylabel('y values')
    plt.legend()
    plt.title('y vs x')
    plt.grid(True, alpha=0.3)
    plt.savefig("plot_interp_pivots.png")

    return
