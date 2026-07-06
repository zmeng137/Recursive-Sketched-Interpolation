import time as tm
import numpy as np


def _right_canonicalize(cores):
    """Right-canonicalize a TT/MPO so every core (except the first) has
    orthonormal rows over all but its left bond.

    Works for cores of any rank with the convention that axis 0 is the left
    bond and the last axis is the right bond (rank-3 MPS ``(l, d, r)`` and
    rank-4 MPO ``(l, d_out, d_in, r)`` are both handled). The right-to-left
    sweep splits each core as ``core = L *_bond B`` via an LQ factorization and
    absorbs ``L`` into the right bond of the left neighbour, leaving the
    represented tensor/operator unchanged (it is a pure gauge transform). The
    input list is not mutated; a new list of cores is returned.

    Right-canonical inputs make the subsequent left-to-right zip-up truncation
    near-optimal: the singular values it ranks become the true Schmidt values
    of the bipartition. Without this the keep/discard decision is gauge-blind
    and accuracy degrades by orders of magnitude (see
    docs/hadamard_accuracy_zipup_vs_rsi.md).
    """
    cores = [c.copy() for c in cores]
    for k in range(len(cores) - 1, 0, -1):
        c = cores[k]
        left = c.shape[0]
        mat = c.reshape(left, -1)                  # (left, rest)
        # LQ via QR of the conjugate transpose: mat^dag = Q R.
        Q, R = np.linalg.qr(mat.conj().T)          # Q:(rest, k), R:(k, left)
        B = Q.conj().T.reshape((Q.shape[1],) + c.shape[1:])  # (k, *mid, right)
        L = R.conj().T                             # (left, k)
        cores[k] = B
        # Absorb L into the right (last) bond of the left neighbour.
        nb = cores[k - 1]
        cores[k - 1] = np.tensordot(nb, L, axes=([nb.ndim - 1], [0]))
    return cores


# Zip-up MPO-MPS contraction
def HadamardTT_zipup(mpo, mps, max_rank, eps=0.0, canonicalize=True, verbose=True):
    """Apply an MPO to an MPS via the zip-up algorithm.

    The MPO-MPS contraction is carried out site by site in a single
    left-to-right sweep. After each site a truncated SVD splits off the new
    (left-canonical) output core and the remaining ``S @ Vt`` factor is
    "zipped" into the next site, so the output bond dimension never exceeds
    ``max_rank`` and the full product tensor is never formed.

    For a Hadamard (element-wise) product of two MPS, build ``mpo`` from one
    of them with :func:`utils_mps.mps_to_diag_mpo` (in the caller) and pass
    the other as ``mps``.

    Reference: Stoudenmire & White, *New J. Phys.* 12, 055026 (2010)
    (arXiv:1002.1305).

    Parameters
    ----------
    mpo : list of np.ndarray
        MPO cores with shape ``(wl, d_out, d_in, wr)``.
    mps : list of np.ndarray
        MPS cores with shape ``(ml, d_in, mr)``. Equal length and matching
        physical (``d_in``) dimensions with ``mpo`` are required.
    max_rank : int
        Maximum bond dimension kept at each on-the-fly truncation.
    eps : float, optional
        Relative singular-value cutoff: at each site singular values below
        ``eps * S[0]`` are dropped. ``0`` keeps all values up to
        ``max_rank``.
    canonicalize : bool, optional
        Right-canonicalize ``mpo`` and ``mps`` (right-to-left) before the
        left-to-right zip sweep so the on-the-fly SVD truncations rank true
        Schmidt values. Strongly recommended (default ``True``); without it
        the truncation is gauge-blind and accuracy can drop by several orders
        of magnitude on non-canonical inputs (e.g. a reversed DMRG MPS).
    verbose : bool, optional
        Print the runtime.

    Returns
    -------
    list of np.ndarray
        MPS cores of the result, left-canonical except the last.
    """
    assert len(mpo) == len(mps), "MPO and MPS lengths do not match."
    L = len(mpo)
    dtype = np.result_type(mpo[0].dtype, mps[0].dtype)

    start_t = tm.time()
    if canonicalize:
        mpo = _right_canonicalize(mpo)
        mps = _right_canonicalize(mps)
    out = [None] * L
    # Carried bond tensor connecting the processed left block (reduced bond
    # r_prev) to the still-open MPO/MPS bonds (wl, ml). Trivial at the left
    # boundary.
    M = np.ones((1, 1, 1), dtype=dtype)

    for k in range(L):
        W = mpo[k]                       # (wl, d_out, d_in, wr)
        A = mps[k]                       # (ml, d_in, mr)
        _, d_out, d_in, wr = W.shape
        _, d_in2, mr = A.shape
        assert d_in == d_in2, f"Physical dimension mismatch at site {k}"
        r_prev = M.shape[0]

        # T[r_prev, d_out, wr, mr] = sum_{wl,ml,d_in} M . W . A
        T = np.tensordot(M, W, axes=([1], [0]))        # (r_prev, ml, d_out, d_in, wr)
        T = np.tensordot(T, A, axes=([1, 3], [0, 1]))  # (r_prev, d_out, wr, mr)

        if k == L - 1:
            # Last site: open bonds collapse (wr == mr == 1); no truncation.
            out[k] = T.reshape(r_prev, d_out, wr * mr)
            break

        mat = T.reshape(r_prev * d_out, wr * mr)
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        if eps > 0:
            r = min(int(np.sum(S > eps * S[0])), max_rank)
        else:
            r = min(len(S), max_rank)
        r = max(r, 1)

        out[k] = U[:, :r].reshape(r_prev, d_out, r)
        # Zip S @ Vt into the next site.
        M = (S[:r, None] * Vt[:r, :]).reshape(r, wr, mr)

    if verbose:
        print(f"Runtime of the zip-up MPO-MPS product: {(tm.time() - start_t) * 1000:.2f} ms.")
    return out
