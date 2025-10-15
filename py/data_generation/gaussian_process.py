import os
import sys
import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gaussian_parameter import components
from utils import save_quantics_tensor_hdf5, load_quantics_tensor_hdf5, convert_1d_to_quantics_tensor, convert_quantics_tensor_to_1d

# Single ND Gaussian function
def gaussian_nd(*coords, **params):
    """
    N-dimensional Gaussian function.
    
    Parameters:
    *coords: coordinate arrays (x1, x2, x3, ..., xn)
    **params: keyword arguments containing:
        - mu_1, mu_2, ..., mu_n: means for each dimension
        - sigma_1, sigma_2, ..., sigma_n: standard deviations for each dimension
        - corr: correlation matrix (optional, n×n array, default=identity)
    
    Example for 4D:
        gaussian_nd(x1, x2, x3, x4, mu_1=0, mu_2=0, mu_3=0, mu_4=0,
                    sigma_1=1, sigma_2=1, sigma_3=1, sigma_4=1,
                    corr=np.eye(4))
    """
    n = len(coords)
    
    # Extract means
    mu = np.array([params[f'mu_{i+1}'] for i in range(n)])
    
    # Extract sigmas
    sigma = np.array([params[f'sigma_{i+1}'] for i in range(n)])
    
    # Extract correlation matrix (default to identity)
    corr = params.get('corr', np.eye(n))
    corr = np.array(corr)
    
    # Construct covariance matrix: Σ = D * R * D
    D = np.diag(sigma)
    cov = D @ corr @ D
    
    # Determinant of covariance matrix
    det = np.linalg.det(cov)
    
    # Check for valid covariance matrix
    if det <= 0:
        raise ValueError("Covariance matrix must be positive definite")
    
    # Normalization factor
    norm = 1 / ((2 * np.pi)**(n/2) * np.sqrt(det))
    
    # Inverse of covariance matrix
    cov_inv = np.linalg.inv(cov)
    
    # Calculate deviations from mean
    dx = np.array([coords[i] - mu[i] for i in range(n)])
    
    # Reshape for matrix multiplication
    original_shape = dx.shape[1:]
    dx_flat = dx.reshape(n, -1)
    
    # Compute quadratic form
    temp = cov_inv @ dx_flat
    quadratic_form = np.sum(dx_flat * temp, axis=0)
    quadratic_form = quadratic_form.reshape(original_shape)
    
    # Exponent
    exponent = -0.5 * quadratic_form
    
    return norm * np.exp(exponent)


# Mixed ND Gaussian function
def mixed_gaussian_nd(*coords, components):
    """
    Mixed N-dimensional Gaussian function (sum of multiple Gaussians).
    
    Parameters:
    *coords: coordinate arrays (x1, x2, x3, ..., xn)
    components: list of dictionaries, each containing:
        - 'weight': mixing weight (should sum to 1)
        - 'mu_1', 'mu_2', ..., 'mu_n': means for each dimension
        - 'sigma_1', 'sigma_2', ..., 'sigma_n': standard deviations for each dimension
        - 'corr': correlation matrix (optional, n×n array, default=identity)
    
    Example for 4D:
        components = [
            {
                'weight': 0.6,
                'mu_1': 0, 'mu_2': 0, 'mu_3': 0, 'mu_4': 0,
                'sigma_1': 1, 'sigma_2': 1, 'sigma_3': 1, 'sigma_4': 1,
                'corr': np.eye(4)
            },
            {
                'weight': 0.4,
                'mu_1': 2, 'mu_2': 2, 'mu_3': 2, 'mu_4': 2,
                'sigma_1': 0.5, 'sigma_2': 0.5, 'sigma_3': 0.5, 'sigma_4': 0.5
            }
        ]
        result = mixed_gaussian_nd(x1, x2, x3, x4, components=components)
    """
    result = np.zeros_like(coords[0])
    
    n = len(coords)
    
    for comp in components:
        weight = comp['weight']
        
        # Extract all mu and sigma parameters
        params = {}
        for i in range(n):
            params[f'mu_{i+1}'] = comp[f'mu_{i+1}']
            params[f'sigma_{i+1}'] = comp[f'sigma_{i+1}']
        
        # Extract correlation matrix if present
        if 'corr' in comp:
            params['corr'] = comp['corr']
        
        result += weight * gaussian_nd(*coords, **params)
    
    return result

power = 6
grid_size = 2**power
x1 = np.linspace(-5, 5, grid_size)
x2 = np.linspace(-5, 5, grid_size)
x3 = np.linspace(-5, 5, grid_size)
x4 = np.linspace(-5, 5, grid_size)

# Create 4D meshgrid
X1, X2, X3, X4 = np.meshgrid(x1, x2, x3, x4, indexing='ij')

# Now pass the meshgrids
mix_gauss_4d = mixed_gaussian_nd(X1, X2, X3, X4, components=components[3])

# Print some statistics
print(f"Maximum value: {np.max(mix_gauss_4d):.6f}")
print(f"Minimum value: {np.min(mix_gauss_4d):.6f}")
print(f"Sum of all values: {np.sum(mix_gauss_4d):.6f}")

# Reshape 2D to 1D function (row-major by default)
Gaussfunction_1d = mix_gauss_4d.flatten()
print(f"1D shape (simple flatten): {Gaussfunction_1d.shape}")

# Verify round-trip conversion
Zfunction_2d_recovered = Gaussfunction_1d.reshape(mix_gauss_4d.shape)
print(f"\nRound-trip conversion successful: {np.allclose(mix_gauss_4d, Zfunction_2d_recovered)}")

# Quantics tensor generation
qtensor_from_func1d = convert_1d_to_quantics_tensor(Gaussfunction_1d, 4 * power)
func1d_from_qtensor = convert_quantics_tensor_to_1d(qtensor_from_func1d)

print(f"Error - Function 1D -> quantics tensor -> Function 1D: {tl.norm(func1d_from_qtensor - Gaussfunction_1d)}")

# Save the data
filepath = "/home/zmeng5/QTTM/datasets/qtensor_gaussian/mix4d_gaussian_1.hdf5"
save_quantics_tensor_hdf5(qtensor_from_func1d, filepath)

qtensor_new, metadata = load_quantics_tensor_hdf5(filepath)
print(f"Difference between the real tensor and hdf5-loaded tensor: {tl.norm(qtensor_from_func1d - qtensor_new)}")
