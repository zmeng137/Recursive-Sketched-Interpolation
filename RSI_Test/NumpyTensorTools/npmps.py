import numpy as np
from ncon import ncon
import sys, copy
import scipy as sp
import plot_utility_jax as pltut

def SRC(mpo, mps, max_bond_dim=None, oversampling=0, verbose=False):
    """
    Successive Randomized Compression (SRC) algorithm for MPO-MPS product
    Based on the paper "Successive randomized compression: A randomized algorithm for the compressed MPO-MPS product"
    
    Args:
        mpo: Matrix Product Operator (list of tensors)
        mps: Matrix Product State (list of tensors)
        max_bond_dim: Maximum bond dimension for output MPS
        oversampling: Oversampling parameter for randomized algorithm
        verbose: Whether to print progress information
        
    Returns:
        Compressed MPS representing MPO applied to MPS
    """
    # Check inputs
    check_MPO_links(mpo)
    check_MPS_links(mps)
    mpo = copy.copy(mpo)
    mps = copy.copy(mps)
    n = len(mps)
    d = mps[0].shape[1]  # Physical dimension
    
    if max_bond_dim is None:
        max_bond_dim = mps[0].shape[0] * mpo[0].shape[0]  # Default to maximum possible value
    
    # Working bond dimension (with oversampling)
    chi_work = min(max_bond_dim + oversampling, d * max_bond_dim)
    
    if verbose:
        print(f"SRC: n={n}, d={d}, max_bond_dim={max_bond_dim}, chi_work={chi_work}")
    
    # Step 1: Generate random matrices
    # Generate Omega^{(1)}, ..., Omega^{(n-1)}, each of size d × chi_work
    Omega_list = []
    for i in range(n-1):
        Omega = np.random.randn(d, chi_work)
        Omega_list.append(Omega)
    
    # Step 2: Compute intermediate tensors C^{(1)}, ..., C^{(n-1)}
    C_tensors = [None] * (n-1)
    
    # Compute C^{(1)}
    # C^{(1)}(a,b,c) = sum_{d,e} Omega^{(1)}(a,d) * H^{(1)}(d,b,e) * psi^{(1)}(e,c)
    mpo[0] = np.squeeze(mpo[0], axis=0)
    mps[0] = np.squeeze(mps[0], axis=0)
    C1 = ncon([Omega_list[0], mpo[0], mps[0]], 
                [(1, -1), (1, 2, -2), (2, -3)])
    C_tensors[0] = C1
    
    # Compute C^{(2)} to C^{(n-1)}
    for i in range(1, n-1):
        # C^{(i)}(a,b,c) = sum_{d,e,f,g} C^{(i-1)}(a,d,e) * Omega^{(i)}(a,f) * 
        #                   H^{(i)}(d,f,b,g) * psi^{(i)}(e,g,c)
        C_prev = C_tensors[i-1]
        C_current = ncon([C_prev, Omega_list[i], mpo[i], mps[i]], 
                        [(-1, 1, 2), (3, -2), (1, 3, 4, -3), (2, 4, -4)])
        id = C_current.shape[0]
        idx = np.arange(id)
        C_tensors[i] = C_current[idx, idx, :, :]
    
    # Step 3: Process sites from right to left
    result_mps = [None] * n
    S_tensors = [None] * (n+1)  # S^{(j)} tensors
    
    # Process last site (n)
    # Compute Y^{(n)} = (H|ψ⟩) · Ω^{(1:n-1)}
    mpo[n-1] = np.squeeze(mpo[n-1], axis=-1)
    mps[n-1] = np.squeeze(mps[n-1], axis=-1)
    Y_n = ncon([C_tensors[n-2], mpo[n-1], mps[n-1]], 
                [(-1, 1, 2), (1, -2, 3), (2, 3)])
    Y_n = Y_n.transpose(1, 0)  # (col, row)
    # QR decomposition: Y^{(n)} = η^{(n)} · R^{(n)}
    Q_n, R_n = np.linalg.qr(Y_n, mode='reduced')
    # Form the tensor for the last site
    Q_n = Q_n.transpose(1, 0)
    result_mps[n-1] = Q_n 
    # Compute S^{(n)}
    S_n = ncon([np.conj(result_mps[n-1]), mpo[n-1], mps[n-1]], 
                [(-1, 1), (-2, 1, 2), (-3, 2)])
    S_tensors[n] = S_n

    
    # Process sites n-1 to 2
    for j in range(n-1, 1, -1):  # j = n-1, n-2, ..., 2
        if verbose:
            print(f"Processing site {j}")
        
        # Compute Y^{(j)}
        # Y^{(j)} = sum C^{(j-1)} * H^{(j)} * psi^{(j)} * S^{(j+1)}
        Y_j = ncon([C_tensors[j-2], mpo[j-1], mps[j-1], S_tensors[j+1]], 
                    [(-1, 1, 2), (1, -2, 3, 4), (2, 3, 5), (-3, 4, 5)])
        Y_j = Y_j.transpose(2, 1, 0)
        Y_j = Y_j.reshape(-1, chi_work)
        # QR decomposition
        Q_j, R_j = np.linalg.qr(Y_j, mode='reduced')
        # Form the tensor for the j-th site
        Q_j = Q_j.reshape(-1, d, Q_j.shape[1]).transpose(2, 1, 0)
        result_mps[j-1] = Q_j
        
        # Compute S^{(j)}

        # Sj is B
        S_j = ncon([np.conj(result_mps[j-1]), mpo[j-1], mps[j-1], S_tensors[j+1]], 
                    [(-1, 1, 2), (-2, 1, 4, 5), (-3, 4, 6), (2, 5, 6)])
        S_tensors[j] = S_j
    
    # Step 4: Process first site
    # Compute η^{(1)}
    eta_1 = ncon([mpo[0], mps[0], S_tensors[2]], 
                [(-1, 2, 3), (2, 4), (-2, 3, 4)])
    result_mps[0] = eta_1.reshape(1, d, -1)
    result_mps[-1] = result_mps[-1].reshape(-1, d, 1)
        
    if verbose:
        bond_dims = [t.shape[2] for t in result_mps]
        print(f"SRC completed. Final bond dimensions: {bond_dims}")
        
    return result_mps

def inds_to_num (inds, dx, shift):
    bstr = ''
    for i in inds:
        bstr += str(i)
    return pltut.bin_to_dec (bstr, dx, shift)

def dataarr_to_mps(data_arr, n_sites, phy_dim, cutoff=0.0):
    """This function make the data_arr be the left canonical mps but the last tensor
    of mps is not normalized.

    Variables.
    -----------
    data_arr: 1D ndarray
            The data you prepare to translate to mps.
    n_sites: int
            How many site you want to set.
    phy_dim: int
            Physical bond dimension for each site.
    cutoff: float64
            Ignore less than singular value from SVD decomposed in numpy.linagl.svd. 

    The returned list of tensot has legs ``vL, i, vR`` (as one of the Bs) and Ss list resverve the singular 
    value of right bond on site[i]."""
    
    assert len(data_arr) == phy_dim**n_sites, "Array length must be 2^n_sites."
    # N = n_sites//2
    # print(data_arr)
    # che_matrix = data_arr.reshape(int(2**N),int(2**N)).T
    # for i in range(0,2**n_sites):
    #     inds = format(i, '016b')  
    #     print(inds)
    #     xinds, yinds = inds[:N], inds[N:]
    #     x = inds_to_num (xinds, dx=1, shift = 0)
    #     y = inds_to_num (yinds, dx=1, shift = 0)
    #     print(x,y)
    #     data_arr[i] = che_matrix[x][y]
    # print(data_arr)
    # initialized MPS
    mps = []
    #Ss = []
    #Ss.append(np.ones([1], dtype=float))
    
    # rebuild MPS
    current_shape = data_arr.reshape([phy_dim] * n_sites) #python inds
    current_shape = np.transpose(current_shape, np.arange(current_shape.ndim)[::-1]) #our inds
    #create virtual bond dimension array
    bond_dim = [i for i in range(0,n_sites)]
    for i in range(n_sites):
        # create the left mps:[1, phy_dim, bond_dim[0]]
        #print('current_shape.shape=',current_shape.shape,'i=',i)
        if i == 0:
            u, s, vh = np.linalg.svd(current_shape.reshape(phy_dim, -1), full_matrices=False)
            bond_dim[i] = np.sum(s > cutoff)
            u = u[:, :bond_dim[i]]
            s = s[:bond_dim[i]]
            vh = vh[:bond_dim[i], :]
            mps.append(u.reshape(1, phy_dim, bond_dim[i]))
            #Ss.append(s)
        elif i == n_sites-1:
            #mps.append(current_shape.reshape(bond_dim[i-1], phy_dim,-1))
            u, s, vh = np.linalg.svd(current_shape.reshape(bond_dim[i-1]*phy_dim, -1), full_matrices=False)
            bond_dim[i] = np.sum(s > cutoff)
            u = u[:, :bond_dim[i]]
            s = s[:bond_dim[i]]
            vh = vh[:bond_dim[i], :]
            u = u@s
            mps.append(u.reshape(bond_dim[i-1], phy_dim, bond_dim[i]))
        #     #Ss.append(s[:, np.newaxis] * vh)
        else:
            # create bulk tensor :[bond_dim, phy_dim, bond_dim]
            u, s, vh = np.linalg.svd(current_shape.reshape(bond_dim[i-1]*phy_dim, -1), full_matrices=False)
            bond_dim[i] = np.sum(s > cutoff)
            u = u[:, :bond_dim[i]]
            s = s[:bond_dim[i]]
            vh = vh[:bond_dim[i], :]
            mps.append(u.reshape(bond_dim[i-1], phy_dim, bond_dim[i]))
            #Ss.append(s)
            #LB_mps = s #the boundary tensor of the most right side of left canonical mps
        # prepare the iterated s[:, np.newaxis] * vh = u
        current_shape = (s[:, np.newaxis] * vh).reshape([-1] + [phy_dim] * (n_sites - i -1))
        #print('current_shape.shape=',current_shape.shape,'i=',i)
        #print('current_shape.shape=',current_shape.shape,'i=',i)
    return mps

def reshape_tensor_to_matrix (T, rowrank):
    dims1 = T.shape[:rowrank]
    dims2 = T.shape[rowrank:]
    d1 = np.prod(dims1)
    d2 = np.prod(dims2)
    T = T.reshape((d1,d2))
    return T, dims1, dims2

def truncate_svd2 (T, rowrank, toRight, maxdim=100000000, mindim=1, cutoff=0.):
    T, ds1, ds2 = reshape_tensor_to_matrix (T, rowrank)

    #U, S, Vh = np.linalg.svd (T, full_matrices=False)
    try:
        U, S, Vh = np.linalg.svd (T, full_matrices=False)
    except np.linalg.LinAlgError:
        #np.save('svd_matrix.npy', T)
        rr = np.random.uniform(low=1e-14, high=2e-14, size=T.shape)
        T = T+rr
        U, S, Vh = np.linalg.svd (T, full_matrices=False)
    # S is in decending order

    # Normalized S
    Sn = S / np.sum(S)

    ii_keep = np.full(len(S), False)
    ii_keep[:maxdim] = True
    ii_keep = np.logical_and (ii_keep, (Sn >= cutoff))

    # Truncation error
    ii_trunc = np.logical_not (ii_keep)
    terr = np.sum(S[ii_trunc])

    ii_keep[:mindim] = True

    U, S, Vh = U[:,ii_keep], S[ii_keep], Vh[ii_keep,:]
    S = np.diag(S)


    if toRight:
        A = U.reshape(*ds1,-1)
        B = (S @ Vh).reshape(-1,*ds2)
    else:
        A = (U @ S).reshape(*ds1,-1)
        B = Vh.reshape(-1,*ds2)
    return A, B, terr

def qr_decomp (T, rowrank, toRight):
    T, ds1, ds2 = reshape_tensor_to_matrix (T, rowrank)
    if toRight:
        Q, R = sp.linalg.qr (T, mode='economic')
        Q = Q.reshape(*ds1,-1)
    else:
        R, Q = sp.linalg.rq (T, mode='economic')
        Q = Q.reshape(-1,*ds2)
    return Q, R

#       |
#      (A)
#       |
#       | i'
#   ---(h)---
#       | i
#       |
#      (B)
#       |
#
def applyLocal_mpo_tensor (h, A=None, B=None):
    re = ncon([h,A], ((-1,-2,1,-4), (1,-3)))
    re = ncon([re,B], ((-1,1,-3,-4), (-2,1)))
    return re

#       |
#      (A)
#       |
#       | i'
#   ---(h)---
#       | i
#       |
#      (B)
#       |
#
def applyLocal_mpo (H, A=None, B=None):
    res = []
    for i in range(len(H)):
        h = applyLocal_mpo_tensor (H[i], A, B)
        res.append(h)
    return res

#       |
#      (U)
#       |
#       | i'
#   ---(h)---
#       | i
#       |
#     (Udag)
#       |
#
def applyLocalRot_mpo (H, U):
    Udag = np.conj(np.transpose(U))
    res = []
    for i in range(len(H)):
        h = applyLocal_mpo_tensor (H[i], U, Udag)
        res.append(h)
    return res

#       |
#     (op)
#       |
#   ---(A)---
#
def applyLocal_mps (mps, op):
    res = []
    for i in range(len(mps)):
        A = mps[i]
        A = ncon([A,op], ((-1,1,-3), (-2,1)))
        res.append(A)
    return res

#======================== MPS #========================
# For each tensor, the order of index is (left, i, right)

def check_MPS_links (mps):
    assert mps[0].shape[0] == mps[-1].shape[-1] == 1
    for i in range(len(mps)):
        assert mps[i].ndim == 3
        if i != 0:
            assert mps[i-1].shape[-1] == mps[i].shape[0]
            assert mps[i].shape[1] == mps[0].shape[1]
            assert mps[i].dtype == mps[0].dtype

def random_MPS(N, phydim, seed, vdim=1, dtype=np.complex128):
    mps = []
    np.random.seed(seed)
    for i in range(N):
        if i == 0:
            mps.append (np.random.rand (1, phydim, vdim).astype(dtype) + 1j*np.random.rand (1, phydim, vdim).astype(dtype))
        elif i == N-1:
            mps.append (np.random.rand (vdim, phydim, 1).astype(dtype) + 1j*np.random.rand (vdim, phydim, 1).astype(dtype))
        else:
            mps.append (np.random.rand (vdim, phydim, vdim).astype(dtype)+ 1j*np.random.rand  (vdim, phydim, vdim).astype(dtype))
    return mps

def to_canonical_form (mps, oc):
    mps = copy.copy(mps)
    for i in range(oc):
        mps[i], R = qr_decomp (mps[i], rowrank=2, toRight=True)
        mps[i+1] = ncon((R,mps[i+1]), ((-1,1),(1,-2,-3)))
    for i in range(len(mps)-1,oc,-1):
        mps[i], R = qr_decomp (mps[i], rowrank=1, toRight=False)
        mps[i-1] = ncon((mps[i-1],R), ((-1,-2,1),(1,-3)))
    return mps

def check_canonical (mps, oc):
    for i in range(oc):
        L = ncon([np.conj(mps[i]),mps[i]], ((1,2,-1),(1,2,-2)))
        assert np.linalg.norm(L - np.eye(L.shape[0])) < 1e-12
    for i in range(len(mps)-1,oc,-1):
        R = ncon([np.conj(mps[i]),mps[i]], ((-1,2,1),(-21,2,1)))
        assert np.linalg.norm(R - np.eye(R.shape[0])) < 1e-12

def svd_compress_MPS (mps, maxdim=sys.maxsize, cutoff=0.):
    mps = copy.copy(mps)
    # Left to right
    for p in range(len(mps)-2):
        #
        #         |            |
        #      ---O---  =>  ---O------O---
        mps[p], R, err = truncate_svd2 (mps[p], rowrank=2, toRight=True, maxdim=maxdim, cutoff=cutoff)
        #                -2
        #             1   |
        #    -1 ---O------O--- -3
        mps[p+1] = ncon((R,mps[p+1]), ((-1,1),(1,-2,-3)))

    # Right to left
    for p in range(len(mps)-1,0,-1):
        #
        #         |                   |
        #      ---O---  =>  ---O------O---
        L, mps[p], err = truncate_svd2 (mps[p], rowrank=1, toRight=False, maxdim=maxdim, cutoff=cutoff)
        #         -2
        #          |   1
        #    -1 ---O------O--- -3
        mps[p-1] = ncon((mps[p-1],L), ((-1,-2,1),(1,-3)))
    return mps

def compress_MPS (mps, maxdim=sys.maxsize, cutoff=0.):
    N = len(mps)
    #
    #        2 ---
    #            |
    #   R =      o
    #            |
    #        1 ---
    #
    Rs = [None for i in range(N+1)]
    Rs[-1] = np.ones((1,1))
    for i in range(N-1, -1, -1):
        Rs[i] = ncon([Rs[i+1],mps[i],np.conj(mps[i])], ((1,2), (-1,3,1), (-2,3,2)))

    #norm = Rs[0].reshape(-1)
    #assert len(norm) == 1
    #norm = norm[0]

    #
    #          2
    #          |
    #   rho =  o
    #          |
    #          1
    #
    rho = ncon([Rs[1],mps[0],np.conj(mps[0])], ((1,2), (-1,-2,1), (-3,-4,2)))
    rho = rho.reshape((rho.shape[1], rho.shape[3]))

    #         1
    #         |
    #    0 x--o-- 2
    #
    evals, U = np.linalg.eigh(rho)
    U = U.reshape((1,*U.shape))
    res = [U]

    #
    #        ---- 2
    #        |
    #   L =  |
    #        |
    #        ---- 1
    #
    L = np.array([1.]).reshape((1,1))
    for i in range(1,N):
        #
        #         2---(U)-- -2
        #         |    |
        #   L =  (L)   3
        #         |    |
        #         1---(A)--- -1
        #
        L = ncon([L,mps[i-1],np.conj(U)], ((1,2), (1,3,-1), (2,3,-2)))

        #
        #          -- -1
        #          |
        #   A =    |       -2
        #          |        |
        #         (L)--1--(mps)--- -3
        #
        A = ncon([L,mps[i]], ((1,-1), (1,-2,-3)))

        #
        #         -3 --(A)--2--
        #               |     |
        #              -4     |
        #   rho =            (R)
        #              -2     |
        #               |     |
        #         -1 --(A)--1--
        #
        rho = ncon([Rs[i+1],A,np.conj(A)], ((1,2), (-1,-2,1), (-3,-4,2)))
        d = rho.shape
        rho = rho.reshape((d[0]*d[1], d[2]*d[3]))

        #         1
        #         |
        #     0 --o-- 2
        #
        evals, U  = np.linalg.eigh(rho)
        # truncate by dimension
        DD = min(maxdim, mps[i].shape[2])
        U = U[:,-DD:]
        evals = evals[-DD:]
        # truncate by cutoff
        iis = (evals > cutoff)
        U = U[:,iis]
        #
        U = U.reshape((d[2],d[3],U.shape[1]))

        if i != N-1:
            res.append(U)
        else:
            res.append(np.conj(A))
    return res

def sum_MPS_tensor (T1, T2):
    res = np.zeros((T1.shape[0]+T2.shape[0], T1.shape[1], T1.shape[2]+T2.shape[2]))
    res[:T1.shape[0],:,:T1.shape[2]] = T1
    res[T1.shape[0]:,:,T1.shape[2]:] = T2
    return res

def inner_MPS (mps1, mps2):
    assert len(mps1) == len(mps2)
    res = ncon([mps1[0], np.conj(mps2[0])], ((1,2,-1), (1,2,-2)))
    for i in range(1,len(mps1)):
        res = ncon([res,mps1[i],np.conj(mps2[i])], ((1,2), (1,3,-1), (2,3,-2)))
    res = res.reshape(-1)
    assert len(res) == 1
    return res[0]

def normalize_MPS (mps):
    mps = copy.copy(mps)
    mps[0] = mps[0] / (inner_MPS(mps,mps))**0.5
    return mps

def MPS_dims (mps):
    dims = [mps[i].shape[2] for i in range(len(mps))]
    return dims

def change_dtype (mpso, dtype):
    for i in range(len(mpso)):
        mpso[i] = mpso[i].astype(dtype)
    return mpso

def conj (mps):
    mps = copy.copy(mps)
    for i in range(len(mps)):
        mps[i] = np.conj(mps[i])
    return mps

#======================== MPO ========================
# For each tensor, the order of index is (left, i', i, right)
# An MPO also has a left and a right tensor

def check_MPO_links (mpo):
    assert mpo[0].shape[0] == mpo[-1].shape[-1] == 1
    assert mpo[0].shape[1] == mpo[0].shape[2]
    for i in range(len(mpo)):
        assert mpo[i].ndim == 4
        if i != 0:
            assert mpo[i-1].shape[-1] == mpo[i].shape[0]
            assert mpo[i].shape[1] == mpo[i].shape[2] == mpo[0].shape[1]
            assert mpo[i].dtype == mpo[0].dtype

def MPO_dims (mpo):
    dims = [mpo[i].shape[3] for i in range(len(mpo))]
    return dims

def exact_apply_MPO (mpo, mps):
    assert len(mpo) == len(mps)
    check_MPO_links(mpo)
    check_MPS_links(mps)

    mpo = copy.copy(mpo)

    A1 = ncon([mps[0], mpo[0]], ((-1,1,-4),(-2,-3,1,-5)))
    dl1,dl2,di,dr1,dr2 = A1.shape
    A1 = A1.reshape((1,di,dr1,dr2))
    res = []
    for i in range(1,len(mps)):
        A2 = ncon([mps[i], mpo[i]], ((-1,1,-4),(-2,-3,1,-5)))
        AA = ncon([A1,A2], ((-1,-2,1,2),(1,2,-3,-4,-5)))
        dl,di1,di2,dr1,dr2 = AA.shape
        AA = AA.reshape((dl*di1, di2*dr1*dr2))
        try:
            U, S, Vh = np.linalg.svd (AA, full_matrices=False)
        except np.linalg.LinAlgError:
        #np.save('svd_matrix.npy', T)
            rr = np.random.uniform(low=1e-14, high=2e-14, size=AA.shape)
            AA = AA+rr
            U, S, Vh = np.linalg.svd (AA, full_matrices=False)
        #U, S, Vh = np.linalg.svd (AA, full_matrices=False)
        A = (U*S).reshape((dl,di1,-1))
        A1 = Vh.reshape((-1,di2,dr1,dr2))

        res.append(A)
    dl1,di,dr1,dr2 = A1.shape
    A = A1.reshape((dl1,di,1))
    res.append(A)
    return res

def identity_MPO (N, phydim):
    As = [np.identity(phydim).reshape((1,phydim,phydim,1)) for i in range(N)]
    return As

def absort_L (mpo, L):
    mpo = copy.copy(mpo)
    shape = (1, *mpo[0].shape[1:])
    mpo[0] = ncon([mpo[0],L], ((1,-1,-2,-3),(1,))).reshape(shape)
    return mpo

def absort_R (mpo, R):
    mpo = copy.copy(mpo)
    shape = (*mpo[-1].shape[:3], 1)
    #print("R.shape",R.shape)
    #print("mpo[-1].shape",mpo[-1].shape)
    mpo[-1] = ncon([mpo[-1],R], ((-1,-2,-3,1),(1,))).reshape(shape)
    return mpo

def absort_LR (mpo, L, R):
    mpo = absort_L (mpo, L)
    mpo = absort_R (mpo, R)
    return mpo

def direct_product_2MPO (mpo1, mpo2):
    assert type(mpo1) == list and type(mpo2) == list
    return mpo1 + mpo2

def purify_MPO (mpo, cutoff=0.):
    # Make the MPO like an MPS by splitting the physical indices
    mps = []
    for A in mpo:
        A1, A2, err = truncate_svd2 (A, 2, toRight=True, cutoff=cutoff)
        mps += [A1, A2]
    return mps

def svd_compress_MPO (mpo, maxdim=sys.maxsize, cutoff=0.):
    mpo = copy.copy(mpo)
    # Left to right
    for p in range(len(mpo)-2):
        #
        #         |            |
        #      ---O---  =>  ---O------O---
        #         |            |
        mpo[p], R, err = truncate_svd2 (mpo[p], rowrank=3, toRight=True, maxdim=maxdim, cutoff=cutoff)
        #                -2
        #             1   |
        #    -1 ---O------O--- -4
        #                 |
        #                -3
        mpo[p+1] = ncon((R,mpo[p+1]), ((-1,1),(1,-2,-3,-4)))

    # Right to left
    for p in range(len(mpo)-1,0,-1):
        #
        #         |                   |
        #      ---O---  =>  ---O------O---
        #         |                   |
        L, mpo[p], err = truncate_svd2 (mpo[p], rowrank=1, toRight=False, maxdim=maxdim, cutoff=cutoff)
        #         -2
        #          |   1
        #    -1 ---O------O--- -4
        #          |
        #         -3
        mpo[p-1] = ncon((mpo[p-1],L), ((-1,-2,-3,1),(1,-4)))
    return mpo

def compress_MPO (mpo, cutoff):
    # Make the MPO like an MPS by splitting the physical indices
    mps = purify_MPO (mpo)
    mps2 = compress_MPS (mps, cutoff=cutoff)    # <mps2|mps2> = 1
    c = inner_MPS (mps,mps2)
    mps2[0] *= c

    # Back to MPO
    res = []
    for i in range(0,len(mps),2):
        A = ncon([mps2[i],mps2[i+1]], ((-1,-2,1), (1,-3,-4)))
        res.append(A)
    return res

def sum_MPO_tensor (T1, T2):
    assert T1.ndim == T2.ndim == 4
    # Set dtype
    assert T1.dtype in (int, float, complex) and T2.dtype in (int, float, complex)
    dtype = max(T1.dtype, T2.dtype)

    res = np.zeros((T1.shape[0]+T2.shape[0], T1.shape[1], T1.shape[2], T1.shape[3]+T2.shape[3]), dtype=dtype)
    res[:T1.shape[0],:,:,:T1.shape[3]] = T1
    res[T1.shape[0]:,:,:,T1.shape[3]:] = T2
    return res

def sum_2MPO (mpo1, mpo2):
    assert type(mpo1) == list and type(mpo2) == list
    N = len(mpo1)
    assert N == len(mpo2)

    mpo = []
    for n in range(N):
        A = sum_MPO_tensor (mpo1[n], mpo2[n])
        mpo.append(A)

    L = np.array([1,1])
    R = np.array([1,1])
    mpo = absort_LR (mpo, L, R)
    return mpo

def inner_MPO (mps1, mps2, mpo):
    assert len(mps1) == len(mps2) == len(mpo)
    res = np.ones((1,1,1))
    for i in range(len(mps1)):
        res = ncon([res,mps1[i],mpo[i],np.conj(mps2[i])], ((1,2,3), (1,4,-1), (2,5,4,-2), (3,5,-3)))
    res = res.reshape(-1)
    assert len(res) == 1
    return res[0]

def MPO_contract_all (mpo):
    res = mpo[0]
    for A in mpo[1:]:
        ds1 = [-i for i in range(1,len(res.shape))]
        ds2 = [-i+ds1[-1] for i in range(1,len(A.shape))]
        res = ncon([res,A], ((*ds1,1), (1,*ds2)))
    N = len(mpo)
    return res

def MPO_to_matrix (mpo):
    if len(mpo) > 8:
        print('Not support MPO length > 8')
        print(len(mpo))
        raise Exception
    T = MPO_contract_all (mpo)
    N = len(mpo)*2 + 2
    ii1 = range(1,N-1,2)
    ii2 = range(2,N-1,2)
    shape = (0,*list(ii1),*list(ii2),N-1)
    dim1 = [T.shape[i] for i in ii1]
    dim2 = [T.shape[i] for i in ii2]
    dim1 = np.prod(dim1)
    dim2 = np.prod(dim2)
    T = T.transpose(shape).reshape (dim1, dim2)
    return T

# MPO tensor:
#                 (ipr)                   (ipr)                        (ipr)
#                   1                       0                            1
#                   |                       |                            |
#         (k1)  0 --o-- 3 (k2)              o-- 2 (k)           (k)  0 --o
#                   |                       |                            |
#                   2                       1                            2
#                  (i)                     (i)                          (i)
#
#
#                   2                       0                            2
#                   |                       |                            |
#           T1  0 --o-- 4                   o-- 2                    0 --o
#                   |                       |                            |
#           T2  1 --o-- 5                   o-- 3                    1 --o
#                   |                       |                            |
#                   3                       1                            3
#
#
#                   1                       0                           1
#                   |                       |                           |
#               0 --o-- 2                   o-- 1                   0 --o
#
def prod_MPO_tensor (T1, T2):
    di = T2.shape[2]
    dipr = T1.shape[1]
    dk1 = T1.shape[0] * T2.shape[0]
    dk2 = T1.shape[3] * T2.shape[3]

    T = ncon ([T1,T1], ((-1,-3,1,-5), (-2,1,-4,-6)))
    T = T.reshape ((dk1,dipr,di,dk2))
    return T

def H1_to_two_particles (H1):
    I = np.array([[1.,0.],[0.,1.]])

    N = len(H1)
    H = []
    for i in range(N):
        HIi = ncon([H1[i],I], ((-1,-2,-4,-6), (-3,-5)))
        IHi = ncon([H1[i],I], ((-1,-3,-5,-6), (-2,-4)))
        d = H1[i].shape
        HIi = HIi.reshape((d[0], d[1]*2, d[2]*2, d[3]))
        IHi = IHi.reshape((d[0], d[1]*2, d[2]*2, d[3]))
        H.append (sum_MPO_tensor (HIi, IHi))

    L = np.ones(2)
    R = np.ones(2)
    H = absort_LR (H, L, R)
    return H

def get_H_2D (H_1D):
    N = len(H_1D)
    H_I = identity_MPO (N, 2)
    H1 = direct_product_2MPO (H_1D, H_I)
    H2 = direct_product_2MPO (H_I, [np.transpose(mpo, (3, 1, 2, 0)) for mpo in H_1D][::-1])
    H = sum_2MPO (H1, H2)
    return H


def get_H_3D (H_1D):
    N = len(H_1D)
    H_I = identity_MPO (N, 2)
    H_2I = identity_MPO (2*N, 2)
    H1 = direct_product_2MPO (H_1D, H_2I)
    H2 = direct_product_2MPO (H_I, H_1D)
    H2 = direct_product_2MPO (H2, H_I)
    H3 = direct_product_2MPO (H_2I, H_1D)
    H = sum_2MPO (H1, H2)
    H = sum_2MPO (H, H3)
    return H

