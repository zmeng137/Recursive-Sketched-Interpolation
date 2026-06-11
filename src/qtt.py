import os
import sys
import copy
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import npmps
import differential as df
from ncon import ncon

def slice_first_modes(arr, indices):
    # Slice the first len(indices) modes with given indices
    slicing = tuple(indices) + tuple(slice(None) for _ in range(arr.ndim - len(indices)))
    return arr[slicing]  # Use square brackets, not parentheses

# Slice tensor: T[:,J]
def slice_last_modes(arr, indices):
    # Slice the last len(indices) modes with given indices
    slicing = tuple(slice(None) for _ in range(arr.ndim - len(indices))) + tuple(indices)
    return arr[slicing] 

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


# ===================== MPS/MPO QTT tools (merged from qtt_tools.py) =====================

def grow_site_2D_0th (psi,dtype = np.complex128):
    assert len(psi) % 2 == 0
    psi = copy.copy(psi)
    t0 = np.array([[[1.],[1.]]],dtype=dtype)
    psi[0:0] = [t0]
    psi[len(psi):len(psi)] = [t0]
    npmps.check_MPS_links(psi)
    return psi

def grow_site_2D_1th (psi,maxdim,dtype = np.complex128):
    assert len(psi) % 2 == 0
    nsite = len(psi)
    psi = copy.copy(psi)
    psi_0th = grow_site_2D_0th(psi)
    psi_0th = copy.copy(psi_0th)
    bulk_x_mpo = np.zeros ((2,2,2,2),dtype=dtype)
    bulk_x_mpo[0,:,:,0] = df.I
    bulk_x_mpo[1,:,:,0] = df.sp
    bulk_x_mpo[1,:,:,1] = df.sm

    bulk_y_mpo = np.zeros ((2,2,2,2),dtype=dtype)
    bulk_y_mpo[0,:,:,0] = df.I
    bulk_y_mpo[1,:,:,0] = df.sm
    bulk_y_mpo[1,:,:,1] = df.sp

    left_xbond_tenop = np.zeros ((1,2,2,2),dtype=dtype)
    left_xbond_tenop[0,:,:,0] = df.su+0.5*df.sd
    left_xbond_tenop[0,:,:,1] = 0.5*df.sd
    right_ybond_tenop = bulk_y_mpo[:,:,:,0:1]

    op = [left_xbond_tenop]+[bulk_x_mpo]*(nsite//2) + [bulk_y_mpo]*(nsite//2) + [right_ybond_tenop]
    psi_1th = []
    nsite_ext = len(psi_0th)
    for i in range(nsite_ext):
        ten = ncon ([op[i],psi_0th[i]], ((-1,-3,3,-4), (-2,3,-5)))
        arr = np.shape(ten)
        ten = ten.reshape((arr[0]*arr[1], arr[2], arr[3]*arr[4]))
        psi_1th[i:i] = [ten]
    npmps.check_MPS_links(psi_1th)
    psi_1th = npmps.compress_MPS (psi_1th, maxdim=maxdim)
    return psi_1th

def kill_site_2D(psi, maxdim,dtype = np.complex128):
    assert len(psi) % 2 == 0
    nsite = len(psi)
    kill_ten = np.array([1/2,1/2],dtype=dtype)
    psi = copy.copy(psi)
    left_ten= ncon ([kill_ten,psi[0]], ((1,), (-1,1,-2)))
    right_ten= ncon ([kill_ten,psi[-1]], ((1,), (-1,1,-2)))
    kill_psi = [psi[i] for i in range(1,int(len(psi)-1))]
    kill_psi[0] = ncon ([left_ten,kill_psi[0]], ((-1,1), (1,-2,-3)))
    kill_psi[-1]=ncon ([right_ten,kill_psi[-1]], ((1,-3), (-1,-2,1)))
    npmps.check_MPS_links(kill_psi)
    #kill_psi = npmps.compress_MPS (kill_psi, maxdim=maxdim)
    return kill_psi


def MPS_tensor_to_MPO_tensor (A):
    assert A.ndim == 3
    T = np.zeros((A.shape[0],A.shape[1],A.shape[1],A.shape[2]), dtype=A.dtype)
    for i in range(A.shape[0]):
        for j in range(A.shape[2]):
            ele = A[i,:,j]
            T[i,:,:,j] = np.diag(ele)
    return T

def MPS_to_MPO (mps):
    npmps.check_MPS_links (mps)

    mpo = []
    for A in mps:
        T = MPS_tensor_to_MPO_tensor (A)
        mpo.append(T)
    return mpo

def normalize_MPS_by_integral (mps, x1, x2, Dim):
    mps = copy.copy(mps)
    c = npmps.inner_MPS (mps, mps)
    mps[0] = mps[0] / c**0.5

    N = len(mps)//Dim
    Ndx = 2**N
    dx = (x2-x1)/Ndx

    for d in range(Dim):
        i = d*N
        mps[i] = mps[i] / dx**0.5
    return mps

def sum_elements (mps):
    A = np.array([1,1]).reshape((1,2,1))
    mps2 = [A for i in range(len(mps))]
    return npmps.inner_MPS (mps, mps2)
