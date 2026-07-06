import time as tm
import numpy as np

from multiply_zipup import HadamardTT_zipup
from utils_mps import (
    left_orthonormalize,
    right_orthonormalize,
    mpo_mps_absorb_left,
    mpo_mps_absorb_right,
    mpo_mps_local_solve,
)

# Variational (fitting) MPO-MPS contraction
def HadamardTT_variational(mpo, mps, max_rank, n_sweeps=4, init=None,
                           tol=1e-12, verbose=True):
    """Apply an MPO to an MPS by variational fitting.

    An output MPS ``M`` of bond dimension at most ``max_rank`` is optimized
    to minimize ``|| M - W @ psi ||`` by sweeping back and forth and, at each
    site, replacing the local core with the projection of the exact product
    onto that site (the global optimum for the local least-squares problem
    when ``M`` is kept in mixed canonical form). This is the standard
    DMRG-style fit / "variational MPO application".

    For a Hadamard (element-wise) product of two MPS, build ``mpo`` from one
    of them with :func:`utils_mps.mps_to_diag_mpo` (in the caller) and pass
    the other as ``mps``.

    The initial guess defaults to the cheap :func:`HadamardTT_zipup` result,
    which the sweeps then refine; pass ``init`` to override it (e.g. a zip-up
    result already computed by the caller, avoiding a redundant zip-up). The
    output bond profile is fixed by the initial guess.

    Parameters
    ----------
    mpo : list of np.ndarray
        MPO cores with shape ``(wl, d_out, d_in, wr)``.
    mps : list of np.ndarray
        MPS cores with shape ``(ml, d_in, mr)``.
    max_rank : int
        Maximum bond dimension (used for the default zip-up initial guess).
    n_sweeps : int, optional
        Number of full back-and-forth sweeps.
    init : list of np.ndarray, optional
        Initial guess MPS. Defaults to the zip-up product.
    tol : float, optional
        Early-stop threshold on the relative change of the fit norm between
        sweeps.
    verbose : bool, optional
        Print the runtime and per-sweep fit.

    Returns
    -------
    list of np.ndarray
        MPS cores of the result, right-canonical except site 0.
    """
    assert len(mpo) == len(mps), "MPO and MPS lengths do not match."
    L = len(mpo)
    dtype = np.result_type(mpo[0].dtype, mps[0].dtype)
    psi = [c.astype(dtype) for c in mps]

    if init is None:
        M = HadamardTT_zipup(mpo, mps, max_rank, verbose=False)
    else:
        M = [c.copy() for c in init]
    M = [c.astype(dtype) for c in M]

    start_t = tm.time()

    # Right-canonicalize M so the orthogonality center sits at site 0.
    for k in range(L - 1, 0, -1):
        B, Lmat = right_orthonormalize(M[k])
        M[k] = B
        M[k - 1] = np.tensordot(M[k - 1], Lmat, axes=([2], [0]))

    # Right environments RE[k] cover sites k+1 .. L-1 of <M | W | psi>.
    RE = [None] * L
    RE[L - 1] = np.ones((1, 1, 1), dtype=dtype)
    for k in range(L - 2, -1, -1):
        RE[k] = mpo_mps_absorb_right(RE[k + 1], M[k + 1], mpo[k + 1], psi[k + 1])

    boundary = np.ones((1, 1, 1), dtype=dtype)
    LE = [None] * L
    LE[0] = boundary

    sweep_times = []
    prev_fit = None
    for sweep in range(n_sweeps):
        sweep_t0 = tm.time()
        # left to right
        for k in range(L):
            Mnew = mpo_mps_local_solve(LE[k], RE[k], mpo[k], psi[k])
            if k < L - 1:
                Q, _ = left_orthonormalize(Mnew)
                M[k] = Q
                LE[k + 1] = mpo_mps_absorb_left(LE[k], Q, mpo[k], psi[k])
            else:
                M[k] = Mnew

        # right to left
        for k in range(L - 1, -1, -1):
            Mnew = mpo_mps_local_solve(LE[k], RE[k], mpo[k], psi[k])
            if k > 0:
                B, _ = right_orthonormalize(Mnew)
                M[k] = B
                RE[k - 1] = mpo_mps_absorb_right(RE[k], B, mpo[k], psi[k])
            else:
                M[k] = Mnew

        sweep_times.append(tm.time() - sweep_t0)

        # With M canonical at site 0, ||M[0]|| equals the overlap <M|W|psi> and increases monotonically toward ||W psi||.
        fit = float(np.linalg.norm(M[0]))
        if verbose:
            print(f"  variational sweep {sweep + 1}/{n_sweeps}: "
                  f"fit = {fit:.10e}, runtime = {sweep_times[-1] * 1000:.2f} ms")
        if prev_fit is not None and abs(fit - prev_fit) <= tol * max(fit, 1.0):
            break
        prev_fit = fit

    if verbose:
        # The output bond profile is fixed at the target max_rank by the initial
        # guess, so every sweep runs at full bond. For complexity-scaling
        # comparisons the relevant cost is a single full-bond sweep (steady
        # state), not the total over all sweeps plus the one-time setup
        # (canonicalization + environment build); report the cheapest sweep.
        one_sweep = min(sweep_times) if sweep_times else float("nan")
        print(f"Runtime of the variational Hadamard product: "
              f"total {(tm.time() - start_t) * 1000:.2f} ms; "
              f"single full-bond sweep {one_sweep * 1000:.2f} ms.")
    return M
