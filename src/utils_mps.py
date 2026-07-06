import h5py
import numpy as np

def mps_all_contract(mps):
    result = np.array([[1.0]])

    for core in mps:
        result = result @ core

    return result[0, 0]

def mps_norm(mps):
    n = len(mps)
    mps_copy = mps.copy()
    
    # Contract each core with all-ones vector
    for i in range(n):
        # Sum over physical dimension: contract with all-ones
        mps_copy[i] = mps_copy[i][:,0,:] + mps_copy[i][:,1,:] + mps_copy[i][:,2,:]  #np.sum(mps_copy[i], axis=1)  # Shape: (r_i, r_{i+1})

    norm = mps_all_contract(mps_copy)

    return norm

def tt_inner(A, B):
    """Euclidean inner product <A|B> of two TTs/MPS, contracted bond by bond.

    ``A`` and ``B`` are lists of cores with shape ``(r_left, d, r_right)`` and
    matching length / physical dimensions. The first argument is conjugated, so
    ``<A|A>`` is the squared 2-norm. Cost is O(L * d * chi^3); no dense tensor is
    formed.
    """
    assert len(A) == len(B), "TT lengths do not match."
    dtype = np.result_type(A[0].dtype, B[0].dtype)
    E = np.ones((1, 1), dtype=dtype)          # (a_bond, b_bond)
    for Ak, Bk in zip(A, B):
        T = np.tensordot(E, np.conj(Ak), axes=([0], [0]))   # (b_bond, d, a_right)
        E = np.tensordot(T, Bk, axes=([0, 1], [0, 1]))      # (a_right, b_right)
    return E[0, 0]


def tt_difference(A, B):
    """TT representing ``A - B`` (cores stacked block-diagonally).

    Both are lists of ``(r_left, d, r_right)`` cores of equal length / physical
    dims; the result has bond ``r^A + r^B`` on each internal cut. Used to measure
    residual norms without the catastrophic cancellation of expanding
    ``<A-B|A-B>``.
    """
    assert len(A) == len(B), "TT lengths do not match."
    L = len(A)
    dtype = np.result_type(A[0].dtype, B[0].dtype)
    C = []
    for k in range(L):
        Ak, Bk = A[k], B[k]
        al, d, ar = Ak.shape
        bl, _, br = Bk.shape
        if k == 0:
            # First core: concat along the right bond; fold the minus sign in here.
            C.append(np.concatenate([Ak, -Bk], axis=2).astype(dtype))
        elif k == L - 1:
            # Last core: concat along the left bond.
            C.append(np.concatenate([Ak, Bk], axis=0).astype(dtype))
        else:
            c = np.zeros((al + bl, d, ar + br), dtype=dtype)
            c[:al, :, :ar] = Ak
            c[al:, :, ar:] = Bk
            C.append(c)
    return C


def tt_norm_l2(tt):
    """2-norm ``||tt||`` via a left-to-right QR sweep (backward stable).

    Working with amplitudes (and rank-revealing QR) rather than squared overlaps
    keeps full precision even when ``tt`` is a near-cancelling difference whose
    norm is many orders below its cores -- where ``sqrt(<tt|tt>)`` would lose
    ~half its digits. ``tt`` is a list of ``(r_left, d, r_right)`` cores.
    """
    dtype = tt[0].dtype
    R = np.ones((1, 1), dtype=dtype)
    for core in tt:
        _, d, r2 = core.shape
        M = np.tensordot(R, core, axes=([1], [0]))      # (rows, d, r2)
        rows = M.shape[0]
        _, R = np.linalg.qr(M.reshape(rows * d, r2))
    return float(np.linalg.norm(R))


def tt_rel_error(approx, exact, exact_norm=None):
    """Relative 2-norm error ``||approx - exact|| / ||exact||``, dense-free.

    The residual norm is computed from the explicit difference TT
    (:func:`tt_difference`) via the stable QR-based :func:`tt_norm_l2`, so it
    resolves errors far below the ~1e-8 floor of overlap-expansion formulas
    without ever forming the full dense tensor. Pass ``exact_norm =
    tt_norm_l2(exact)`` to skip recomputing the denominator.
    """
    den = exact_norm if exact_norm is not None else tt_norm_l2(exact)
    return tt_norm_l2(tt_difference(approx, exact)) / den


def mps_diagonal(mps_square):
    n = len(mps_square)
    mps_copy = mps_square.copy()
    for i in range(n):
        mps_copy[i] = np.sum(mps_copy[i], axis=1)  # Shape: (r_i, r_{i+1})

    energy = 0.0
    for j in range(n-1):
        mps_copy_ = mps_copy.copy()
        mps_copy_[j] = mps_square[j][:,0,:] - mps_square[j][:,2,:]  
        mps_copy_[j+1] = mps_square[j+1][:,0,:] - mps_square[j+1][:,2,:]
    
        contrib = mps_all_contract(mps_copy_)
        energy = energy + contrib

    return energy


# ---------------------------------------------------------------------------
# MPS / MPO helpers for the zip-up and variational Hadamard-product methods
# ---------------------------------------------------------------------------
def mps_to_diag_mpo(mps):
    """Reinterpret an MPS as a diagonal MPO.

    Each MPS core ``A[a, i, b]`` becomes an MPO core
    ``W[a, i, j, b] = A[a, i, b] * delta(i, j)`` of shape
    ``(r_left, d_out, d_in, r_right)``. Applying this MPO to a second MPS
    contracts ``j`` against that MPS' physical index, which is exactly the
    element-wise (Hadamard) product.
    """
    mpo = []
    for A in mps:
        d = A.shape[1]
        W = np.einsum('aib,ij->aijb', A, np.eye(d, dtype=A.dtype))
        mpo.append(W)
    return mpo


def left_orthonormalize(core):
    """Left-orthonormalize a core. Returns (Q, R) with core = Q *_bond R and
    Q reshaped to (r1, d, k) satisfying Q^dag Q = I over (r1, d)."""
    r1, d, r2 = core.shape
    Q, R = np.linalg.qr(core.reshape(r1 * d, r2))
    return Q.reshape(r1, d, Q.shape[1]), R


def right_orthonormalize(core):
    """Right-orthonormalize a core. Returns (B, L) with core = L *_bond B and
    B reshaped to (k, d, r2) satisfying B B^dag = I over (d, r2)."""
    r1, d, r2 = core.shape
    Q, R = np.linalg.qr(core.reshape(r1, d * r2).conj().T)   # M^dag = Q R
    B = Q.conj().T.reshape(Q.shape[1], d, r2)
    L = R.conj().T                                            # (r1, k)
    return B, L


def mpo_mps_absorb_left(LE, Mc, Wc, Ac):
    """Grow the left environment of <M | W | psi> past one site.

    LE:(a, wl, ml), Mc:(a, d_out, a'), Wc:(wl, d_out, d_in, wr),
    Ac:(ml, d_in, mr)  ->  (a', wr, mr).
    """
    T = np.tensordot(LE, np.conj(Mc), axes=([0], [0]))   # (wl, ml, d_out, a')
    T = np.tensordot(T, Wc, axes=([0, 2], [0, 1]))       # (ml, a', d_in, wr)
    T = np.tensordot(T, Ac, axes=([0, 2], [0, 1]))       # (a', wr, mr)
    return T


def mpo_mps_absorb_right(RE, Mc, Wc, Ac):
    """Grow the right environment of <M | W | psi> past one site.

    RE:(a', wr, mr), Mc:(a, d_out, a'), Wc:(wl, d_out, d_in, wr),
    Ac:(ml, d_in, mr)  ->  (a, wl, ml).
    """
    T = np.tensordot(np.conj(Mc), RE, axes=([2], [0]))   # (a, d_out, wr, mr)
    T = np.tensordot(T, Wc, axes=([1, 2], [1, 3]))       # (a, mr, wl, d_in)
    T = np.tensordot(T, Ac, axes=([1, 3], [2, 1]))       # (a, wl, ml)
    return T


def mpo_mps_local_solve(LE, RE, Wc, Ac):
    """Project the exact MPO-MPS product onto the local site.

    LE:(a, wl, ml), RE:(b, wr, mr), Wc:(wl, d_out, d_in, wr),
    Ac:(ml, d_in, mr)  ->  optimal core (a, d_out, b).
    """
    T = np.tensordot(LE, Wc, axes=([1], [0]))        # (a, ml, d_out, d_in, wr)
    T = np.tensordot(T, Ac, axes=([1, 3], [0, 1]))   # (a, d_out, wr, mr)
    T = np.tensordot(T, RE, axes=([2, 3], [1, 2]))   # (a, d_out, b)
    return T


def readh5_mps(filePath):
    with h5py.File(filePath, "r") as f:
        num_sites = f["num_sites"][()]
        energy = f["energy"][()]
        energy_diag = f["energy_diag"][()]
        
        mps = []
        for i in range(1, num_sites + 1):
            tensor = f[f"tensor_{i}"][:]
            shape = f[f"shape_{i}"][:]
            mps.append(tensor)
            
            if i == 1:
                print(f"Site {i} (first): shape {tensor.shape} = (physical={shape[0]}, right_bond={shape[1]})")
            elif i == num_sites:
                print(f"Site {i} (last): shape {tensor.shape} = (left_bond={shape[0]}, physical={shape[1]})")
            else:
                print(f"Site {i}: shape {tensor.shape} = (left_bond={shape[0]}, physical={shape[1]}, right_bond={shape[2]})")
        
        print(f"\nGround state energy: {energy}")

    mps.reverse()
    mps[0] = mps[0].reshape(1,mps[0].shape[0],mps[0].shape[1])
    mps[-1] = mps[-1].reshape(mps[-1].shape[0],mps[-1].shape[1],1)

    mps_rank = [mps[0].shape[0]] + [mps[k].shape[2] for k in range(len(mps)-1)] + [mps[-1].shape[2]]
    print(f"\nMPS rank: {mps_rank}\n")

    return mps, num_sites, energy, energy_diag