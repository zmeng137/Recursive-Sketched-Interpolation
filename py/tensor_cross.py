import numpy as np
import random as rd
import tensorly as tl

from interpolation import interpolative_prrldu, cur_prrldu
from rank_revealing import prrldu

# Slice tensor: T[I,:]
def slice_first_modes(arr, indices):
    # Slice the first len(indices) modes with given indices
    slicing = tuple(indices) + tuple(slice(None) for _ in range(arr.ndim - len(indices)))
    return arr[slicing]  # Use square brackets, not parentheses

# Slice tensor: T[:,J]
def slice_last_modes(arr, indices):
    # Slice the last len(indices) modes with given indices
    slicing = tuple(slice(None) for _ in range(arr.ndim - len(indices))) + tuple(indices)
    return arr[slicing] 

# Stably merge the cross inverse into the left TT-core (How about LU?)
def coreinv_qr(tensor_core, r_pivot):
    # Reshape the TT-core to a matrix
    t_shape = tensor_core.shape
    mrow = t_shape[0] * t_shape[1]
    mcol = t_shape[2]
    core_mat = tl.reshape(tensor_core, [mrow, mcol])
    
    # QR decomposition of the TT-core (for stable inversion, could be replaced by LU)
    Q, _ = np.linalg.qr(core_mat)
    
    # T = Q @ Q[pr,:]^-1. May could be more efficient
    mask = ~np.isin(np.arange(mrow), r_pivot) 
    core_mat[mask] = Q[mask] @ np.linalg.inv(Q[r_pivot, :])
    core_mat[r_pivot, :] = np.identity(mcol)
    
    # Reshape the matrix back to tensor core
    tensor_core = tl.reshape(core_mat, t_shape)
    return tensor_core

# PRRLU-based Tensor-Train Interpolative Decomposition
def TT_IDPRRLDU(tensorX: tl.tensor, r_max: int, eps: float, verbose: int = 0) -> list[tl.tensor]:
    shape = tensorX.shape  # Get the shape of input tensor: [n1, n2, ..., nd]
    dim = len(shape)       # Get the number of dimension
    delta = (eps / np.sqrt(dim - 1)) * tl.norm(tensorX, 2)  # Truncation parameter
    
    W = tensorX        # Copy tensor X -> W
    nbar = W.size      # Total size of W
    r = 1              # Rank r
    ttList = []        # list storing tt factors
    iterlist = list(range(1, dim))  # Create iteration list: 1, 2, ..., d-1
    iterlist.reverse()              # Reverse the iteration list: d-1, ..., 1 
    
    for i in iterlist:
        W = tl.reshape(W, [int(nbar / r / shape[i]), int(r * shape[i])])  # Reshape W       
        C, X, cols, error = interpolative_prrldu(W, cutoff=delta, maxdim=r_max)
        ri = C.shape[1]  # r_i-1 = min(r_max, r_delta_i)
    
        if verbose == 1:
            rerror = tl.norm(C @ X - W, 2) / tl.norm(W, 2)
            print(f"Iteration {i} -- low rank id approximation error = {rerror}")
    
        Ti = tl.reshape(X[0:ri, :], [ri, shape[i], r])
        nbar = int(nbar * ri / shape[i] / r)  # New total size of W
        r = ri             # Renewal r
        W = C[:, 0:ri]     # W = U[..] * S[..]
        ttList.append(Ti)  # Append new factor
    
    T1 = tl.reshape(W, [1, shape[0], r])
    ttList.append(T1)    
    ttList.reverse()
    return ttList

# PRRLU-based (Exact) Tensor-Train CUR Decomposition (Sweep from Left to Right)
def TT_CUR_L2R(tensor: tl.tensor, r_max: int, eps: float, verbose = 1, full_nest = 1):
    shape = tensor.shape  # Get the shape of input tensor: [n1, n2, ..., nd]
    dim = len(shape)      # Get the number of dimension
     
    W = tensor        # Copy tensor X -> W
    nbar = W.size     # Total size of W
    r = 1             # Initial TT-Rank r=1
    TTCore = []       # list storing TT-factors
    TTCore_cc = []    # Tensor-train including intermediate cross cores
    TTRank = [1]      # TT-Rank list
    InterpSet_I = {}  # One-sided nested I(row) index set
    InterpSet_J = {}  # One-sided nested J(col) index set

    for i in range(dim-1):
        curr_dim = shape[i]  # Current dimension
        W = tl.reshape(W, [int(r * curr_dim), int(nbar / r / curr_dim)])  # Reshape W       
        
        # CUR decomposition based on PRRLDU
        r_subset, c_subset, cross, rank, pr, pc = cur_prrldu(W, eps, r_max)
        pr = pr[0:rank]  # Row skeleton 
        pc = pc[0:rank]  # Col skeleton

        # Mapping between r/c selection and tensor index pivots
        if i == 0:
            InterpSet_I[i+1] = np.array(pr).reshape(-1, 1)
        else:
            I = np.empty([rank, i+1])
            prev_I = InterpSet_I[i]
            for j in range(rank):
                p_I_idx = pr[j] // curr_dim
                c_i_idx = pr[j] % curr_dim
                I[j,0:i] = prev_I[p_I_idx]
                I[j,i] = c_i_idx
            InterpSet_I[i+1] = I        
        
        # Get J index set
        if (i == dim - 2):
            InterpSet_J[i+2] = np.array(pc).reshape(-1,1)

        # Append new TT-factor
        Ti = tl.reshape(c_subset, [r, shape[i], rank])
        Ti = coreinv_qr(Ti, pr)
        TTCore.append(Ti)                                          
        TTCore_cc.append(tl.reshape(c_subset, [r, shape[i], rank]))  
        TTCore_cc.append(cross)
        TTRank.append(rank)

        nbar = int(nbar * rank / shape[i] / r)  # New total size of W
        r = rank  # Renewal r
        W = r_subset[0:rank,:]
        
        # Check the nested condition
        if (verbose):
            #print("Checking left fully nesting condition...")
            for ele in np.nditer(W):
                match_idx = np.argwhere(tensor == ele)
                nested_idx = match_idx[0][0:i+1]
                is_present = np.any(np.all(InterpSet_I[i+1] == nested_idx, axis=1))
                if (is_present == False):
                    print("Nested Interpolation error!")
            #print("Done.")
        
    T_last = tl.reshape(W, [r, shape[-1], 1])
    TTCore.append(T_last)    
    TTCore_cc.append(T_last)
    TTRank.append(1)
    
    if (verbose):
        print("Checking if the last two cores' interpolation matches the entries")
        last_c1 = np.empty([TTRank[-2], shape[-1], TTRank[-1]])
        last_c2 = np.empty([TTRank[-3], shape[-2], TTRank[-2]])
        for i in range(TTRank[-2]):
            I_slice = InterpSet_I[dim-1][i].astype(int).tolist()
            last_c1[i,:,0] = slice_first_modes(tensor, I_slice)
        for i in range(TTRank[-3]):
            I_slice = InterpSet_I[dim-2][i].astype(int).tolist()
            for j in range(TTRank[-2]):
                J_slice = InterpSet_J[dim][j].astype(int).tolist()
                temp = slice_first_modes(tensor, I_slice)
                last_c2[i,:,j] = slice_last_modes(temp, J_slice)
        diff1 = tl.norm(last_c1 - TTCore_cc[-1])
        diff2 = tl.norm(last_c2 - TTCore_cc[-3])
        assert diff1 < 1e-14 and diff2 < 1e-14, "Slicing wrong!"
        print("Done.")

    # Site-1 TCI for restoring full nesting
    if (full_nest):
        iterlist = list(range(1, dim-1))  # Create iteration list: 1, 2, ..., d-2
        iterlist.reverse()                # Reverse the iteration list: d-2, ..., 1
        for i in iterlist:
            ccore = TTCore_cc[2 * i]  # Current TT-core
            cshape = ccore.shape      # Core shape
            mat = tl.reshape(ccore, [cshape[0], cshape[1] * cshape[2]], order='F')  # Reshape 3D core to matrix
            _, _, _, _, _, rps, cps, _ = prrldu(mat, 0, cshape[0])  # PRRLU for new pivots
            curr_dim = cshape[1]
            prev_J = InterpSet_J[i+2]
            J = np.empty([cshape[0], dim-i])
            for j in range(cshape[0]):
                p_J_idx = cps[j] // curr_dim
                c_J_idx = cps[j] % curr_dim
                J[j,1:] = prev_J[p_J_idx]
                J[j,0] = c_J_idx
            InterpSet_J[i+1] = J
    
    return TTCore, TTCore_cc, TTRank, InterpSet_I, InterpSet_J

'''
# PRRLU-based (Exact) Tensor-Train CUR Decomposition (Sweep from Right to Left)
def TT_CUR_R2L(tensor: tl.tensor, r_max: int, eps: float, verbose = 1):
    shape = tensor.shape  # Get the shape of input tensor: [n1, n2, ..., nd]
    dim = len(shape)      # Get the number of dimension
     
    W = tensor      # Copy tensor X -> W
    nbar = W.size   # Total size of W
    r = 1           # Initial TT-Rank r=1
    TTCore = []     # list storing TT-factors
    TTCore_cc = []  # Tensor-train including intermediate cross cores
    TTRank = [1]    # TT-Rank list
    InterpSet = {}  # One-sided nested set

    # Mapping between r/c selection and tensor index pivots
    iterlist = list(range(1, dim))  # Create iteration list: 1, 2, ..., d-1
    iterlist.reverse()              # Reverse the iteration list: d-1, ..., 1 
    for i in iterlist:
        curr_dim = shape[i]  # Current dimension
        W = tl.reshape(W, [int(nbar / r / shape[i]), int(r * shape[i])], order='F')  # Reshape W       
        
        r_subset, c_subset, cross_inv, cross, rank, pr, pc = cur_prrldu(W, eps, r_max)
        pr = pr[0:rank]  # Row skeleton 
        pc = pc[0:rank]  # Col skeleton

        # Mapping between r/c selection and tensor index pivots
        pc = np.array(pc)
        if i == dim-1:
            InterpSet[i] = np.array(pc).reshape(-1,1)
        else:
            J = np.empty([rank, dim-i])
            prev_I = InterpSet[i+1]
            for j in range(rank):
                p_I_idx = pc[j] // curr_dim
                c_i_idx = pc[j] % curr_dim
                J[j,1:] = prev_I[p_I_idx]
                J[j,0] = c_i_idx
            InterpSet[i] = J        

        # Append new TT-factor
        Ti = tl.reshape(cross_inv @ r_subset, [rank, shape[i], r], order='F')
        TTCore.append(Ti)                                          
        TTCore_cc.append(tl.reshape(r_subset, [rank, shape[i], r], order='F'))  
        TTCore_cc.append(cross)
        TTRank.append(rank)

        nbar = int(nbar * rank / shape[i] / r)  # New total size of W
        r = rank  # Renewal r
        W = c_subset[:, 0:rank]

       # Check the nested condition
        if (verbose):
            for ele in np.nditer(W):
                match_idx = np.argwhere(tensor == ele)
                nested_idx = match_idx[0][i:]
                is_present = np.any(np.all(InterpSet[i] == nested_idx, axis=1))
                if (is_present == False):
                    print("Nested Interpolation error!")     
        
    T_last = tl.reshape(W, [1, shape[0], r], order='F')
    TTCore.append(T_last)    
    TTCore_cc.append(T_last)
    TTRank.append(1)
    TTCore.reverse()
    TTCore_cc.reverse()
    TTRank.reverse()
    return TTCore, TTCore_cc, TTRank, InterpSet
'''

# Compute inverse of cross matrices and merge them into TT-cores
def cross_inv_merge(TTCore_cross, dimension, order=0, verbose=0):
    if order == 0:
        TTCores = [TTCore_cross[0]]
        for i in range(dimension-1):
            core = TTCore_cross[2*i+2]
            cross = TTCore_cross[2*i+1]
            core_shape0 = core.shape[0]
            core_shape1 = core.shape[1]
            core_shape2 = core.shape[2]    
            cross_inv = np.linalg.inv(cross)
            if verbose == 1:
                rinv = cross_inv @ cross - np.identity(cross.shape[0])
                rerr = tl.norm(rinv) / np.sqrt(cross.shape[0])
                print(f"Cross inverse matrix quality: {1-rerr}")
            core_reshape = core.reshape(core_shape0,-1)
            merge = cross_inv @ core_reshape
            new_core = merge.reshape(core_shape0, core_shape1, core_shape2)
            TTCores.append(new_core)
        return TTCores
    else:
        TTCores = []
        for i in range(dimension-1):
            core = TTCore_cross[2*i]
            cross = TTCore_cross[2*i+1]
            cross_inv = np.linalg.inv(cross)
            if verbose == 1:
                rinv = cross_inv @ cross - np.identity(cross.shape[0])
                rerr = tl.norm(rinv) / np.sqrt(cross.shape[0])
                print(f"Cross inverse matrix quality: {1-rerr}")
            new_core = core @ cross_inv
            TTCores.append(new_core)
        TTCores.append(TTCore_cross[-1])
        return TTCores

# A more stable version of cross_inv_merge (using QR-based inverse)
def cross_inv_merge_stable(TTCore_cross, Pr_set):
    dimension = len(Pr_set) + 1
    TTCores = []
    for i in range(dimension-1):
        core = TTCore_cross[2*i]
        new_core = coreinv_qr(core, Pr_set[i+1])  # Use QR-based inverse
        TTCores.append(new_core)
    TTCores.append(TTCore_cross[-1])
    return TTCores

# Assmeble a single core by interpolation pivots
# TO BE MODIFIED: INVERSE PROBLEM
def single_core_interp_assemble(tensor: tl.tensor, I_interpSet: dict, J_interpSet: dict, TTRank: np.array, core_no: int):
    shape = tensor.shape  # Get the shape of input tensor: [n1, n2, ..., nd]
    dim = len(shape)      # Get the number of dimension 
    assert core_no <= dim-1 and 0 <= core_no, "Query core should be in [0, dim-1]"
    assert len(TTRank) == dim + 1, "Number of TT-Ranks should equal tensor order + 1, i.e., with rank = 1 at boundaries!"
    
    d = core_no
    core = np.empty([TTRank[d], shape[d], TTRank[d+1]])
    cross_mat = np.empty([TTRank[d+1], TTRank[d+1]])

    # Construct TT-cores
    if d == 0:
        assert TTRank[d+1] == len(J_interpSet[2]), "Interpolation set size != Given rank"
        for j in range(TTRank[1]):
            J_slice = J_interpSet[2][j].astype(int).tolist()
            core[0,:,j] = slice_last_modes(tensor, J_slice) 
    elif d == dim-1:
        assert TTRank[d] == len(I_interpSet[d]), "Interpolation set size != Given rank"
        for i in range(TTRank[dim-1]):
            I_slice = I_interpSet[d][i].astype(int).tolist()
            core[i,:,0] = slice_first_modes(tensor, I_slice)
    else:
        for i in range(TTRank[d]):
            I_slice = I_interpSet[d][i].astype(int).tolist()
            for j in range(TTRank[d+1]):
                J_slice = J_interpSet[d+2][j].astype(int).tolist()
                temp = slice_first_modes(tensor, I_slice)
                core[i,:,j] = slice_last_modes(temp, J_slice)

    # Construct cross matrices
    if d != dim-1:
        for i in range(TTRank[d+1]):
            I_slice = I_interpSet[d+1][i].astype(int).tolist()
            for j in range(TTRank[d+1]):
                J_slice = J_interpSet[d+2][j].astype(int).tolist()
                temp = slice_first_modes(tensor, I_slice)
                cross_mat[i,j] = slice_last_modes(temp, J_slice)
        
        cross_inv = np.linalg.inv(cross_mat)
        core = core @ cross_inv
     
    return core

# Assemble TT-Cores by (fully nested) interpolation pivots 
def cross_core_interp_assemble(tensor: tl.tensor, I_interpSet: dict, J_interpSet: dict, TTRank: np.array):
    shape = tensor.shape  # Get the shape of input tensor: [n1, n2, ..., nd]
    dim = len(shape)      # Get the number of dimension 
    assert len(TTRank) == dim + 1, "Number of TT-Ranks should equal tensor order + 1, i.e., with rank = 1 at boundaries!"
    TTCore_cross = []

    # Assmebly of TT-Cores via interpolation sets
    for d in range(dim):
        # Initialize TT-cores and cross matrices
        core = np.empty([TTRank[d], shape[d], TTRank[d+1]])
        cross_mat = np.empty([TTRank[d+1], TTRank[d+1]])
        
        # Construct TT-cores
        if d == 0:
            assert TTRank[d+1] == len(J_interpSet[2]), "Interpolation set size != Given rank"
            for j in range(TTRank[1]):
                J_slice = J_interpSet[2][j].astype(int).tolist()
                core[0,:,j] = slice_last_modes(tensor, J_slice)
            TTCore_cross.append(core) 
        elif d == dim-1:
            assert TTRank[d] == len(I_interpSet[d]), "Interpolation set size != Given rank"
            for i in range(TTRank[dim-1]):
                I_slice = I_interpSet[d][i].astype(int).tolist()
                core[i,:,0] = slice_first_modes(tensor, I_slice)
            TTCore_cross.append(core)
        else:
            for i in range(TTRank[d]):
                I_slice = I_interpSet[d][i].astype(int).tolist()
                for j in range(TTRank[d+1]):
                    J_slice = J_interpSet[d+2][j].astype(int).tolist()
                    temp = slice_first_modes(tensor, I_slice)
                    core[i,:,j] = slice_last_modes(temp, J_slice)
            TTCore_cross.append(core)

        # Construct cross matrices
        if d != dim-1:
            for i in range(TTRank[d+1]):
                I_slice = I_interpSet[d+1][i].astype(int).tolist()
                for j in range(TTRank[d+1]):
                    J_slice = J_interpSet[d+2][j].astype(int).tolist()
                    temp = slice_first_modes(tensor, I_slice)
                    cross_mat[i,j] = slice_last_modes(temp, J_slice)
            TTCore_cross.append(cross_mat)
    return TTCore_cross

# Get the PI tensor (4-order) by slicing the original tensor via interpolation sets  
def PI_4tensor_slicing(tensor, mode1, mode2, I_set, J_set):
    # Initialize the PI tensor
    shape = tensor.shape
    s1 = shape[mode1-1]
    s2 = shape[mode2-1]
    left_rank = 1
    right_rank = 1
    if len(I_set) != 0:
        I_set = I_set.astype(int).tolist()
        left_rank = len(I_set)
    if len(J_set) != 0:
        right_rank = len(J_set)
        J_set = J_set.astype(int).tolist()
    PI_4tensor = np.empty([left_rank, s1, s2, right_rank])

    # Construct the PI tensor
    if I_set == []:
        for j in range(right_rank):
            j_idx = J_set[j]
            PI_4tensor[0,:,:,j] = slice_last_modes(tensor, j_idx)
    elif J_set == []:
        for i in range(left_rank):
            i_idx = I_set[i]
            PI_4tensor[i,:,:,0] = slice_first_modes(tensor, i_idx)
    else:
        for i in range(left_rank):
            i_idx = I_set[i]
            temp = slice_first_modes(tensor, i_idx)
            for j in range(right_rank):
                j_idx = J_set[j]
                PI_4tensor[i,:,:,j] = slice_last_modes(temp, j_idx)
    return PI_4tensor

def TCI_2site(tensor, eps, tt_rmax, interp_I, interp_J, cvg_check = 0):
    # tensor information
    shape = tensor.shape
    dim = len(shape)  # Let's say dim=L, then I is from 1 to L-1, J is from 2 to L
    TTRank = [1] * (dim+1)

    # Initialization
    if interp_I == None or interp_J == None:
        # TODO: Give a reasonable pivot initialization 
        pass
    result_I = interp_I.copy()
    result_J = interp_J.copy()

    # Here we also record the matrix-wise pivots for every core
    pr_set = {}  
    pc_set = {}  

    # TCI sweep iteration
    iter_flag = True
    pre_error = 1
    iter = 1
    while iter_flag:
        # Cross sweep back and forth: left to right
        for l in range(1, dim):
            # Slice tensor 
            I_set = result_I[l-1]
            J_set = result_J[l+2]
            ldim = shape[l-1]
            rdim = len(J_set)
            PI_4tensor_i = PI_4tensor_slicing(tensor, l, l+1, I_set, J_set)
            PI_shape = PI_4tensor_i.shape
            
            # PRRLU cross decomp
            PI_matrix = tl.reshape(PI_4tensor_i, [PI_shape[0]*PI_shape[1], PI_shape[2]*PI_shape[3]])
            _, d, _, _, _, pr, pc, _ = prrldu(PI_matrix, eps, tt_rmax)
            rank = len(d)
            TTRank[l] = rank
            pr = pr[0:rank]
            pc = pc[0:rank]

            # Map pr, pc to I, J
            if l == 1:
                result_I[l] = np.array(pr).reshape(-1, 1)
            else:
                I = np.empty([rank, l])
                prev_I = result_I[l-1]
                for i in range(rank):
                    p_I_idx = pr[i] // ldim
                    c_I_idx = pr[i]  % ldim
                    I[i,0:l-1] = prev_I[p_I_idx]
                    I[i,l-1] = c_I_idx
                result_I[l] = I
            if l == dim-1:
                result_J[l+1] = np.array(pc).reshape(-1,1)
            else:
                J = np.empty([rank, dim-l])
                prev_J = result_J[l+2]
                ####### PROBLEM MAYBE? #####
                for j in range(rank):
                    p_J_idx = pc[j] % rdim  ###
                    c_J_idx = pc[j] // rdim  ###
                    J[j,1:] = prev_J[p_J_idx]
                    J[j,0] = c_J_idx
                result_I[l+1] = J

        # Cross sweep back and forth: right to left
        for l in range(dim, 1, -1):
            I_set = result_I[l-2]
            J_set = result_J[l+1]
            ldim = shape[l-2]
            rdim = len(J_set)
            PI_4tensor_i = PI_4tensor_slicing(tensor, l-1, l, I_set, J_set)
            PI_shape = PI_4tensor_i.shape
            
            # PRRLU cross decomp
            PI_matrix = tl.reshape(PI_4tensor_i, [PI_shape[0]*PI_shape[1], PI_shape[2]*PI_shape[3]])
            _, d, _, _, _, pr, pc, _ = prrldu(PI_matrix, eps, tt_rmax)
            rank = len(d)
            TTRank[l-1] = rank
            pr = pr[0:rank]
            pc = pc[0:rank]
            pr_set[l-1] = pr  # Store row pivots
            pc_set[l] = pc  # Store column pivots

            # Map pr, pc to I, J
            if l == dim:
                result_J[l] = np.array(pc).reshape(-1,1)
            else:
                J = np.empty([rank, dim-l+1])
                prev_J = result_J[l+1]
                for j in range(rank):
                    p_J_idx = pc[j] % rdim  
                    c_J_idx = pc[j] // rdim 
                    J[j,1:] = prev_J[p_J_idx]
                    J[j,0] = c_J_idx
                result_J[l] = J        
            if l == 2:
                result_I[l-1] = np.array(pr).reshape(-1, 1)
            else:
                I = np.empty([rank, l-1])
                prev_I = result_I[l-2]
                for i in range(rank):
                    p_I_idx = pr[i] // ldim
                    c_I_idx = pr[i]  % ldim
                    I[i,0:l-2] = prev_I[p_I_idx]
                    I[i,l-2] = c_I_idx
                result_I[l-1] = I
    
        # TODO... Not a good convergence check
        # Assemble the tensor train and test convergence (error)
        TT_cross = cross_core_interp_assemble(tensor, result_I, result_J, TTRank)
        TT_cores = cross_inv_merge_stable(TT_cross, pr_set)
        if cvg_check == 0:
            recon_t = tl.tt_to_tensor(TT_cores)
            rel_diff = tl.norm(recon_t - tensor) / tl.norm(tensor)
            delta_diff = np.abs(rel_diff - pre_error) / np.abs(pre_error)
            print(f"Iteration {iter} - relative error: {rel_diff}, delta error: {delta_diff}, TTRank: {TTRank}")
            pre_error = rel_diff
            if delta_diff < 1e-8:
                break
        # TODO... A new convergence check
        #else:
             
        iter += 1

    TT_cross = cross_core_interp_assemble(tensor, result_I, result_J, TTRank)
    return TT_cross, TT_cores, TTRank, pr_set, pc_set, result_I, result_J

def TCI_union_two(tensor_f1, interp_I_1, interp_J_1, tensor_f2, interp_I_2, interp_J_2, mode = 0):
    # tensor information
    shape = tensor_f1.shape
    assert shape == tensor_f2.shape, "Two input tensors should be at same size!"
    tensor_g = tensor_f1 * tensor_f2  # Let's first assume we know tensor_f1 and tensor_f2 (we actually know everything we need in this function by only TCI format)
    dim = len(shape)  # Dimension
    TTRank_new = [1]

    # Iteratively combine interpolation sets
    interp_I_new = {}
    interp_J_new = {}
    for d in range(1, dim):
        # Interpolation sets
        I_set_1 = interp_I_1[d]
        J_set_1 = interp_J_1[d+1]
        I_set_2 = interp_I_2[d]
        J_set_2 = interp_J_2[d+1]

        # Union
        I_union = np.unique(np.vstack([I_set_1, I_set_2]), axis=0)
        J_union = np.unique(np.vstack([J_set_1, J_set_2]), axis=0)
        rank_I_union = I_union.shape[0]
        rank_J_union = J_union.shape[0]
        max_rank = min(rank_I_union, rank_J_union)
        TTRank_new.append(max_rank)

        # Check union results
        if rank_I_union == rank_J_union:
            interp_I_new[d] = I_union
            interp_J_new[d+1] = J_union
        else:
            # Assemble cross core            
            interp_I_new[d] = np.empty([max_rank, d]) 
            interp_J_new[d+1] = np.empty([max_rank, dim-d])
            cross_mat = np.empty([rank_I_union, rank_J_union])
            for i in range(rank_I_union):
                I_slice = I_union[i].astype(int).tolist()
                for j in range(rank_J_union):
                    J_slice = J_union[j].astype(int).tolist()
                    temp = slice_first_modes(tensor_g, I_slice)
                    cross_mat[i,j] = slice_last_modes(temp, J_slice)
            # Pivot selection (PRRLDU)
            if mode == 0:
                # PRRLU mode
                _, _, _, _, _, pr, pc, _ = prrldu(cross_mat, 0, max_rank)
                pr = pr[0:max_rank]
                pc = pc[0:max_rank]
                interp_I_new[d] = I_union[pr,:]
                interp_J_new[d+1] = J_union[pc,:]
            else:
                # Random mode
                if max_rank < rank_I_union:
                    selected_indices = np.random.choice(rank_I_union, size=max_rank, replace=False)
                    interp_I_new[d] = I_union[selected_indices,:]
                    interp_J_new[d+1] = J_union
                if max_rank < rank_J_union:
                    selected_indices = np.random.choice(rank_J_union, size=max_rank, replace=False)
                    interp_I_new[d] = I_union
                    interp_J_new[d+1] = J_union[selected_indices,:]
    
    TTRank_new.append(1)
    return interp_I_new, interp_J_new, TTRank_new

# Generate random nested interpolation sets for tensor cross interpolation
def nested_initIJ_gen_rank1(dim):
    # Dict for I/J interpolation
    interp_I = {}
    interp_J = {}

    # Iteration for I interpolation
    interp_I[1] = np.array([[rd.randint(0, 1)]])
    for d in range(2,dim):
        pivot = rd.randint(0, 1)
        interp_I[d] = np.empty([1,d])
        interp_I[d][0,0:d-1] = interp_I[d-1][0]
        interp_I[d][0,d-1] = pivot
    
    # Iteration for J interpolation
    interp_J[dim] = np.array([[rd.randint(0, 1)]])
    for d in range(dim-1, 1, -1):
        pivot = rd.randint(0, 1)
        interp_J[d] = np.empty([1,dim-d+1])
        interp_J[d][0,0] = pivot
        interp_J[d][0,1:] = interp_J[d+1][0]
        
    # Empty boundary interpolation
    interp_I[0] = []
    interp_J[dim+1] = []

    return interp_I, interp_J

# 1-site tensor cross interpolation for nesting condition and rank compression
def TCI_1site(tensor, interp_I, interp_J):

    return