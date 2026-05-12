import numpy as np
import random as rd

from scipy.linalg import hadamard


def hadamard_1d(x):
    """Walsh–Hadamard transform of a 1D array (in-place style)."""
    n = x.shape[0]
    h = 1
    y = x.copy()
    while h < n:
        for i in range(0, n, 2 * h):
            a = y[i:i+h]
            b = y[i+h:i+2*h]
            y[i:i+h] = a + b
            y[i+h:i+2*h] = a - b
        h *= 2
    return y


def srht_sketch_tt_core(G, k, seed=None):
    """
    SRHT sketch of the physical dimension of a TT-core.

    Parameters
    ----------
    G : ndarray, shape (r_prev, d, r_next)
        TT-core
    k : int
        Sketch size (k < d)
    seed : int or None

    Returns
    -------
    Gs : ndarray, shape (r_prev, k, r_next)
        Sketched TT-core
    """
    r_prev, d, r_next = G.shape
    assert (d & (d - 1)) == 0, "d must be a power of two"

    rng = np.random.default_rng(seed)

    # 1. Random sign vector D
    D = rng.choice([-1.0, 1.0], size=d)

    # 2. Subsampling indices
    idx = rng.choice(d, size=k, replace=False)

    # Output
    Gs = np.empty((r_prev, k, r_next), dtype=G.dtype)

    scale = np.sqrt(d / k)

    # Loop over TT-ranks (small)
    for i in range(r_prev):
        for j in range(r_next):
            x = G[i, :, j]

            # Apply SRHT
            x = D * x
            x = hadamard_1d(x)
            Gs[i, :, j] = scale * x[idx]

    return Gs

def generate_srht_sketch(d, k, seed=None):
    """
    Generate SRHT sketch matrix Omega that maps R^d -> R^k.
    
    Parameters:
    -----------
    d : int
        Original dimension
    k : int
        Sketch dimension (k <= d)
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    Omega : np.ndarray
        Sketch matrix of shape (k, d)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Pad d to nearest power of 2 for Hadamard transform
    d_padded = 2**int(np.ceil(np.log2(d)))
    
    # Random diagonal matrix D with ±1 entries
    D = np.diag(np.random.choice([-1, 1], size=d_padded))
    
    # Hadamard matrix (normalized)
    H = hadamard(d_padded) / np.sqrt(d_padded)
    
    # Random sampling matrix P
    # If k > d_padded, sample with replacement
    replace = (k > d_padded)
    sample_indices = np.random.choice(d_padded, size=k, replace=replace)
    P = np.zeros((k, d_padded))
    P[np.arange(k), sample_indices] = 1
    
    # SRHT: Omega = sqrt(d_padded/k) * P * H * D
    scale = np.sqrt(d_padded / k)
    Omega_full = scale * P @ H @ D
    
    # Truncate to original dimension d
    Omega = Omega_full[:, :d]
    
    return Omega

def generate_random_fourier_features(d, k, bandwidth=1.0, seed=None):
    """
    Generate Random Fourier Features sketch optimized for oscillatory data.
    
    This uses random frequencies drawn from a distribution, making it much better
    for capturing oscillatory patterns than standard Gaussian sketches.
    
    Parameters:
    -----------
    d : int
        Original dimension
    k : int
        Sketch dimension
    bandwidth : float
        Controls frequency spread. Higher = capture higher frequencies.
        For sin(10000*x), use bandwidth ~ 10000 or estimate from data
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    Omega : np.ndarray
        Sketch matrix of shape (k, d)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Random frequencies (sample from Gaussian in frequency domain)
    omega = np.random.randn(k, d) * bandwidth
    
    # Random phase shifts
    b = np.random.uniform(0, 2*np.pi, size=k)
    
    # Build sketch: cos(omega * x + b)
    # For discrete case, x_j = j/d
    x = np.arange(d) / d
    
    # Omega[j, m] for sketch dimension j and input dimension m
    Omega = np.zeros((k, d))
    for j in range(k):
        Omega[j, :] = np.cos(omega[j, :] @ x + b[j])
    
    # Normalize
    Omega = Omega * np.sqrt(2.0 / k)
    
    return Omega



def generate_structured_random_sketch(d, k, seed=None):
    """
    Generate a structured random sketch that mixes Gaussian and oscillatory components.
    
    Combines random projections with trigonometric functions to capture both
    smooth and oscillatory features.
    
    Parameters:
    -----------
    d : int
        Original dimension  
    k : int
        Sketch dimension
    seed : int, optional
        Random seed
        
    Returns:
    --------
    Omega : np.ndarray
        Sketch matrix of shape (k, d)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Split sketch into two parts
    k_random = k // 2
    k_trig = k - k_random
    
    # Part 1: Standard Gaussian
    Omega_random = np.random.randn(k_random, d) / np.sqrt(k)
    
    # Part 2: Trigonometric with random frequencies
    x = np.arange(d)
    Omega_trig = np.zeros((k_trig, d))
    
    for i in range(k_trig):
        # Random frequency (covering wide range for oscillatory data)
        freq = np.random.uniform(0.1, min(d/2, 1000))
        phase = np.random.uniform(0, 2*np.pi)
        sign = np.random.choice([-1, 1])
        
        if i % 2 == 0:
            Omega_trig[i, :] = sign * np.cos(2*np.pi*freq*x/d + phase)
        else:
            Omega_trig[i, :] = sign * np.sin(2*np.pi*freq*x/d + phase)
    
    Omega_trig = Omega_trig / np.sqrt(k)
    
    # Combine
    Omega = np.vstack([Omega_random, Omega_trig])
    
    return Omega


def generate_multiscale_sketch(d, k, num_scales=3, seed=None):
    """
    Generate multi-scale sketch that captures features at different frequency scales.
    
    Particularly good for oscillatory data where you don't know the exact frequency.
    Divides sketch dimensions across multiple frequency bands.
    
    Parameters:
    -----------
    d : int
        Original dimension
    k : int
        Sketch dimension
    num_scales : int
        Number of frequency scales to use
    seed : int, optional
        Random seed
        
    Returns:
    --------
    Omega : np.ndarray
        Sketch matrix of shape (k, d)
    """
    if seed is not None:
        np.random.seed(seed)
    
    k_per_scale = k // num_scales
    k_remainder = k % num_scales
    
    Omega = np.zeros((k, d))
    x = np.arange(d)
    
    idx = 0
    for scale in range(num_scales):
        k_scale = k_per_scale + (1 if scale < k_remainder else 0)
        
        # Frequency range for this scale: [2^scale, 2^(scale+1)]
        freq_min = 2**scale
        freq_max = 2**(scale + 1)
        
        for i in range(k_scale):
            freq = np.random.uniform(freq_min, freq_max)
            phase = np.random.uniform(0, 2*np.pi)
            
            if np.random.rand() > 0.5:
                Omega[idx, :] = np.cos(2*np.pi*freq*x/d + phase)
            else:
                Omega[idx, :] = np.sin(2*np.pi*freq*x/d + phase)
            
            idx += 1
    
    # Normalize
    Omega = Omega / np.sqrt(k)
    
    return Omega


def estimate_dominant_frequency(core, axis=1):
    """
    Estimate the dominant frequency in a TT-core along physical dimension.
    
    Parameters:
    -----------
    core : np.ndarray
        TT-core of shape (r_prev, d, r_next)
    axis : int
        Axis to analyze (default 1 for physical dimension)
        
    Returns:
    --------
    freq : float
        Estimated dominant frequency (normalized)
    """
    # Take FFT along physical dimension
    d = core.shape[axis]
    fft_vals = np.fft.fft(core, axis=axis)
    power = np.abs(fft_vals)**2
    
    # Average power across other dimensions
    avg_power = np.mean(power, axis=(0, 2))
    
    # Find dominant frequency
    dominant_idx = np.argmax(avg_power[:d//2])  # Only positive frequencies
    dominant_freq = dominant_idx / d
    
    return dominant_freq, dominant_idx


def sketch_tt_core(core, k, seed=None, method='multiscale', bandwidth='auto'):
    """
    Sketch the physical dimension of a TT-core with frequency-aware methods.
    
    Parameters:
    -----------
    core : np.ndarray
        TT-core of shape (r_prev, d, r_next)
    k : int
        Sketch dimension for physical mode
    seed : int, optional
        Random seed
    method : str
        'rff': Random Fourier Features (need to specify bandwidth)
        'structured': Mix of Gaussian + trigonometric
        'multiscale': Multiple frequency scales (recommended for unknown frequencies)
        'gaussian': Standard Gaussian (baseline)
    bandwidth : float or 'auto'
        For 'rff' method. If 'auto', estimates from data.
        
    Returns:
    --------
    sketched_core : np.ndarray
        Sketched TT-core of shape (r_prev, k, r_next)
    Omega : np.ndarray
        The sketch matrix used (k, d)
    """
    r_prev, d, r_next = core.shape
    
    # Generate sketch matrix based on method
    if method == 'rff':
        if bandwidth == 'auto':
            dominant_freq, _ = estimate_dominant_frequency(core)
            bandwidth = max(dominant_freq * d, 10.0)  # Scale to actual frequency
        Omega = generate_random_fourier_features(d, k, bandwidth, seed)
        
    elif method == 'structured':
        Omega = generate_structured_random_sketch(d, k, seed)
        
    elif method == 'multiscale':
        Omega = generate_multiscale_sketch(d, k, seed=seed)
        
    elif method == 'gaussian':
        if seed is not None:
            np.random.seed(seed)
        Omega = np.random.randn(k, d) / np.sqrt(k)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Apply sketch
    sketched_core = np.einsum('jm,iml->ijl', Omega, core)
    
    return sketched_core, Omega

# Generate a random array with entries from standard normal distribution.
def random_normal_tensor(shape, complex=False):
    if complex:
        # Complex standard normal: real and imaginary parts are N(0, 1/2)
        # So that E[|z|²] = 1
        real_part = np.random.randn(*shape if isinstance(shape, tuple) else (shape,)) / np.sqrt(2)
        imag_part = np.random.randn(*shape if isinstance(shape, tuple) else (shape,)) / np.sqrt(2)
        return real_part + 1j * imag_part
    else:
        # Real standard normal: N(0, 1)
        return np.random.randn(*shape if isinstance(shape, tuple) else (shape,))
    
def random_uniform_tensor(shape, low=0.0, high=1.0, complex=False):
    if complex:
        real_part = np.random.uniform(low, high, size=shape if isinstance(shape, tuple) else (shape,))
        imag_part = np.random.uniform(low, high, size=shape if isinstance(shape, tuple) else (shape,))
        return real_part + 1j * imag_part
    else:
        return np.random.uniform(low, high, size=shape if isinstance(shape, tuple) else (shape,))

def tt_sketching_cache(tt_cores, sketch_tail_no, skdim, seed, verbose=True):
    dim = len(tt_cores)
    shape = [tt_cores[i].shape[1] for i in range(dim)]   
    if verbose:
        print(f"Performing TT random sketching cache... the order-{dim} tensor has shape of {shape}")

    # Formalize the new sketched TT 
    tt_sketched = tt_cores[dim - sketch_tail_no:].copy()
    #rd.seed(seed)
    for i in range(sketch_tail_no):
        core_pos = dim - sketch_tail_no + i
        sk_shape = (shape[core_pos], skdim)
        
        #sk_tensor = np.random.uniform(0, 1, size=sk_shape if isinstance(sk_shape, tuple) else (sk_shape,))
        #sk_tensor = np.random.randn(*sk_shape if isinstance(sk_shape, tuple) else (sk_shape,))
        sk_tensor = np.random.normal(0, 10, size=sk_shape if isinstance(sk_shape, tuple) else (sk_shape,))
        tt_sketched[i] = np.einsum('ijk,jl->ilk', tt_sketched[i], sk_tensor)       

        #tt_sketched[i], Omega = sketch_tt_core(tt_sketched[i], skdim, method='multiscale')
        pass
        

    return tt_sketched
