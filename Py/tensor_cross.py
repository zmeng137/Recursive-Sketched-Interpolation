import numpy as np
import tensorly as tl
from interpolation import interpolative_prrldu, cur_prrldu, cur_prrldu_ninv
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
        r_subset, c_subset, cross_inv, cross, rank, pr, pc = cur_prrldu(W, eps, r_max)
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
        Ti = tl.reshape(c_subset @ cross_inv, [r, shape[i], rank])
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

# Compute inverse of cross matrices and merge them into TT-cores
def cross_inv_merge(TTCore_cross, dimension):
    TTCores = [TTCore_cross[0]]
    for i in range(dimension-1):
        core = TTCore_cross[2*i+2]
        cross = TTCore_cross[2*i+1]
        core_shape0 = core.shape[0]
        core_shape1 = core.shape[1]
        core_shape2 = core.shape[2]    
        cross_inv = np.linalg.inv(cross)
        core_reshape = core.reshape(core_shape0,-1)
        merge = cross_inv @ core_reshape
        new_core = merge.reshape(core_shape0, core_shape1, core_shape2)
        TTCores.append(new_core)
    return TTCores

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

# 1-site tensor cross interpolation for nesting condition and rank compression
def TCI_1site(tensor_train, nested_I, ):

    return

# Sweep and PRRLU -based tensor cross interpolation
def TCI_PRRLU(tensor_func, tol, max_iter):
    
    return

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

def TCI_2site(tensor, tt_rmax, interp_I = None, interp_J = None):
    # tensor information
    shape = tensor.shape
    dim = len(shape)  # Let's say dim=L, then I is from 1 to L-1, J is from 2 to L

    # Initialization
    if interp_I == None or interp_J == None:
        # TODO: Give a reasonable pivot initialization 
        pass

    # Cross sweep back and forth: left to right
    for l in range(1, dim):
        I_set = interp_I[l-1]
        J_set = interp_J[l+2]
        PI_4tensor_i = PI_4tensor_slicing(tensor, l, l+1, I_set, J_set)
        PI_shape = PI_4tensor_i.shape
        PI_matrix = tl.reshape(PI_4tensor_i, [PI_shape[0]*PI_shape[1], PI_shape[2]*PI_shape[3]])
        _, d, _, _, _, pr, pc, _ = prrldu(PI_matrix, 0, tt_rmax)
        pr = pr[0:len(d)]
        pc = pc[0:len(d)]

        # Map pr, pc to I, J
        # ...

    # Cross sweep back and forth: right to left
    for l in range(dim, 1, -1):
        I_set = interp_I[l-1]
        J_set = interp_J[l+2]


    # Assemble the tensor train and test convergence (error)

    return

def TCI_pivot_accum(tensor, interpSet_I, interpSet_J, new_pivot_i):
    
    return

# PROBLEM ALGORITHM!
# Assemble tensor cross interpolation by union of interpolation sets I/J
# Actually the naive union method cannot even work, as there I,J's size at the same mode after union is not same
def TCI_Union(tensor, I1, J1, I2, J2):
    shape = tensor.shape  # Get the shape of input tensor: [n1, n2, ..., nd]
    dim = len(shape)      # Get the number of dimension 
    TTRank = [1]          # TTRank list [1,...,1]
    TTCore_cross = []     # TCI list
    
    # Union of I1 and I2, J1 and J2
    Union_I = {}
    Union_J = {}
    for d in range(dim-1):
        Union_I[d+1] = np.unique(np.vstack([I1[d+1], I2[d+1]]), axis=0)
        Union_J[d+2] = np.unique(np.vstack([J1[d+2], J2[d+2]]), axis=0)
        TTRank.append(Union_I[d+1].shape[0])
    TTRank.append(1)
    # PROBELM!
    # Assemble TT-cores from union interpolation sets
    TTCore_cross = cross_core_interp_assemble(tensor, Union_I, Union_J, TTRank)
    return TTCore_cross