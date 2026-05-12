# plot_utility_jax.py
"""
High-performance JAX-based utilities for evaluating 2D MPS / MPO on binary grids.
This version preserves all original function names (drop-in compatible) and implements
GPU-friendly batching similar to 3D version.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from jax import jit, vmap
from numba import njit, prange

jax.config.update("jax_enable_x64", True)

# ---------------------------
# numba helpers (kept for compatibility)
# ---------------------------
@njit(parallel=True)
def dec_to_bin(dec, N):
    bstr = np.zeros(N, dtype=np.int32)
    for i in prange(N):
        bstr[N - 1 - i] = (dec >> i) & 1
    return bstr

@njit
def bin_array_to_dec(bstr, rescale=1.0, shift=0.0):
    dec = 0
    for i in range(len(bstr)):
        dec += bstr[i] * (2 ** i)
    return dec * rescale + shift

@njit(parallel=True)
def bin_to_dec_list(bstr, rescale=1.0, shift=0.0):
    dec_list = np.zeros(len(bstr), dtype=np.float64)
    for i in prange(len(bstr)):
        dec_list[i] = bin_array_to_dec(bstr[i], rescale, shift)
    return dec_list

# ---------------------------
# Converters
# ---------------------------
def convert_mps_to_jax_arrays(mps_list):
    """Convert MPS list to tuple of JAX arrays (complex128)"""
    return tuple(jnp.array(tensor, dtype=jnp.complex128) for tensor in mps_list)

def convert_mpo_to_jax_arrays(mpo_list):
    """Convert MPO list to tuple of JAX arrays (complex128)"""
    return tuple(jnp.array(tensor, dtype=jnp.complex128) for tensor in mpo_list)

def convert_binary_to_jax_arrays(bxs, bys):
    """Convert lists of binary arrays to device jnp arrays"""
    bxs_jax = jnp.stack([jnp.array([int(bit) for bit in bx], dtype=jnp.int32) for bx in bxs])
    bys_jax = jnp.stack([jnp.array([int(bit) for bit in by], dtype=jnp.int32) for by in bys])
    return bxs_jax, bys_jax

# ---------------------------
# BinaryNumbers iterator
# ---------------------------
class BinaryNumbers:
    """Iterator over all binary strings of length N"""
    def __init__(self, N):
        self.N_num = N
        self.N_dec = 2**N

    def __iter__(self):
        self.dec = 0
        return self

    def __next__(self):
        if self.dec < self.N_dec:
            dec = self.dec
            self.dec += 1
            return dec_to_bin(dec, self.N_num)[::-1]
        else:
            raise StopIteration

# ---------------------------
# Single-element computations
# ---------------------------
@jit
def compute_mps_element_single(mps_tuple, bstr):
    """Compute scalar value for a single binary string (bx + by[::-1])"""
    result = jnp.ones((1,1), dtype=jnp.complex128)
    for i in range(len(mps_tuple)):
        A = mps_tuple[i]
        bi = bstr[i]
        M = A[:, bi, :]
        result = jnp.dot(result, M)
    return result[0,0]

@jit
def compute_mpo_element_single(mpo_tuple, bstr):
    """Compute scalar value for a single binary string for MPO"""
    result = jnp.ones((1,1), dtype=jnp.complex128)
    for i in range(len(mpo_tuple)):
        A = mpo_tuple[i]
        bi = bstr[i]
        M = A[:, bi, bi, :]
        result = jnp.dot(result, M)
    return result[0,0]

# ---------------------------
# Device-side batch helpers
# ---------------------------
@jit
def _build_all_bstrs_from_indices_2D(bxs_j, bys_j, ix, iy):
    """
    Construct 2D batch of bitstrings entirely on device.
    Preserves bx + by[::-1] order.
    """
    bx_batch = bxs_j[ix]            # (B, bits_x)
    by_batch = bys_j[iy][:, ::-1]   # reverse each row
    return jnp.concatenate([bx_batch, by_batch], axis=1)

# ---------------------------
# Backwards-compatible batch interfaces
# ---------------------------
def batch_compute_mps(mps_tuple, bstrs):
    """Legacy interface: batch compute using vmap."""
    @jit
    def batched_func(bstr):
        return compute_mps_element_single(mps_tuple, bstr)
    return vmap(batched_func)(bstrs)

def batch_compute_mpo(mpo_tuple, bstrs):
    @jit
    def batched_func(bstr):
        return compute_mpo_element_single(mpo_tuple, bstr)
    return vmap(batched_func)(bstrs)

# ---------------------------
# Main 2D grid-evaluation functions
# ---------------------------
def get_2D_mesh_eles_mps(mps, bxs, bys, batch_size=2**16, return_complex=True):
    """
    Compute MPS values on a 2D grid defined by lists bxs, bys.
    Preserves bx + by[::-1] order.
    GPU-friendly batching style (like 3D version).
    """
    print("Converting MPS to JAX format...")
    mps_jax = convert_mps_to_jax_arrays(mps)
    nx, ny = len(bxs), len(bys)

    bxs_j = jnp.stack([jnp.array(b, dtype=jnp.int32) for b in bxs])
    bys_j = jnp.stack([jnp.array(b, dtype=jnp.int32) for b in bys])

    total = nx * ny
    result = np.zeros((ny, nx), dtype=np.complex128 if return_complex else np.float64)
    print(f"Total points: {total}, batch_size: {batch_size}")

    # Precompile single-element
    test_bstr = jnp.concatenate([bxs_j[0], bys_j[0][::-1]])
    compute_mps_element_single(mps_jax, test_bstr).block_until_ready()

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        inds = np.arange(start, end, dtype=np.int32)
        ix = inds % nx
        iy = inds // nx

        # Device-side batch bitstrings
        batch_bstrs = _build_all_bstrs_from_indices_2D(bxs_j, bys_j, jnp.array(ix), jnp.array(iy))

        # Batch compute
        batch_vals = vmap(lambda b: compute_mps_element_single(mps_jax, b))(batch_bstrs)
        batch_vals = batch_vals.block_until_ready()
        batch_vals_np = np.array(batch_vals)

        # Fill result
        for n, val in enumerate(batch_vals_np):
            result[int(iy[n]), int(ix[n])] = val if return_complex else abs(val)

        # Optional progress
        if (start // batch_size) % max(1, (total // (10*batch_size))) == 0:
            print(f"  processed {end}/{total}")

    return result

def get_2D_mesh_eles_mpo(mpo, bxs, bys, batch_size=2**16, return_complex=False):
    """
    Compute MPO values on a 2D grid defined by lists bxs, bys.
    Preserves bx + by[::-1] order.
    GPU-friendly batching style (like 3D version).
    """
    print("Converting MPO to JAX format...")
    mpo_jax = convert_mpo_to_jax_arrays(mpo)
    nx, ny = len(bxs), len(bys)

    bxs_j = jnp.stack([jnp.array(b, dtype=jnp.int32) for b in bxs])
    bys_j = jnp.stack([jnp.array(b, dtype=jnp.int32) for b in bys])

    total = nx * ny
    result = np.zeros((ny, nx), dtype=np.complex128 if return_complex else np.float64)
    print(f"Total points: {total}, batch_size: {batch_size}")

    # Precompile single-element
    test_bstr = jnp.concatenate([bxs_j[0], bys_j[0][::-1]])
    compute_mpo_element_single(mpo_jax, test_bstr).block_until_ready()

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        inds = np.arange(start, end, dtype=np.int32)
        ix = inds % nx
        iy = inds // nx

        batch_bstrs = _build_all_bstrs_from_indices_2D(bxs_j, bys_j, jnp.array(ix), jnp.array(iy))

        batch_vals = vmap(lambda b: compute_mpo_element_single(mpo_jax, b))(batch_bstrs)
        batch_vals = batch_vals.block_until_ready()
        batch_vals_np = np.array(batch_vals)

        for n, val in enumerate(batch_vals_np):
            result[int(iy[n]), int(ix[n])] = val if return_complex else abs(val)

        if (start // batch_size) % max(1, (total // (10*batch_size))) == 0:
            print(f"  processed {end}/{total}")

    return result

# ---------------------------
# Optional: single-element access
# ---------------------------
def get_ele_mps(mps, bstr):
    mps_jax = convert_mps_to_jax_arrays(mps)
    bstr_jax = jnp.array([int(bit) for bit in bstr], dtype=jnp.int32)
    return np.array(compute_mps_element_single(mps_jax, bstr_jax))

def get_ele_mpo(mpo, bstr):
    mpo_jax = convert_mpo_to_jax_arrays(mpo)
    bstr_jax = jnp.array([int(bit) for bit in bstr], dtype=jnp.int32)
    return np.array(compute_mpo_element_single(mpo_jax, bstr_jax))
