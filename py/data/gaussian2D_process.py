import os
import sys
import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from the_well.data import WellDataset
from utils import save_quantics_tensor_hdf5, load_quantics_tensor_hdf5, convert_1d_to_quantics_tensor, convert_quantics_tensor_to_1d

from gaussian2D_parameter import components

# Single 2D Gaussian function
def gaussian_2d(x, y, mu_x, mu_y, sigma_x, sigma_y, rho=0):
    """
    Parameters:
    x, y: coordinate arrays
    mu_x, mu_y: means in x and y directions
    sigma_x, sigma_y: standard deviations in x and y directions
    rho: correlation coefficient between x and y
    """
    # Covariance matrix
    sigma_xx = sigma_x**2
    sigma_yy = sigma_y**2
    sigma_xy = rho * sigma_x * sigma_y
    
    # Determinant of covariance matrix
    det = sigma_xx * sigma_yy - sigma_xy**2
    
    # Normalization factor
    norm = 1 / (2 * np.pi * np.sqrt(det))
    
    # Exponent
    dx = x - mu_x
    dy = y - mu_y
    exponent = -0.5 / det * (sigma_yy * dx**2 + sigma_xx * dy**2 - 2 * sigma_xy * dx * dy)
    
    return norm * np.exp(exponent)

# Mixed 2D Gaussian function (sum of multiple Gaussians)
def mixed_gaussian_2d(x, y, components):
    """
    Parameters:
    x, y: coordinate arrays
    components: list of dictionaries, each containing:
        - 'weight': mixing weight (should sum to 1)
        - 'mu_x', 'mu_y': means
        - 'sigma_x', 'sigma_y': standard deviations
        - 'rho': correlation coefficient (optional, default=0)
    """
    result = np.zeros_like(x)
    
    for comp in components:
        weight = comp['weight']
        mu_x = comp['mu_x']
        mu_y = comp['mu_y']
        sigma_x = comp['sigma_x']
        sigma_y = comp['sigma_y']
        rho = comp.get('rho', 0)  # default correlation is 0
        
        result += weight * gaussian_2d(x, y, mu_x, mu_y, sigma_x, sigma_y, rho)
    
    return result

x_dim = 8
y_dim = 8
x = np.linspace(-10, 10, 2 ** x_dim)
y = np.linspace(-10, 10, 2 ** y_dim)
X, Y = np.meshgrid(x, y)

comp = components[1]
    
# Generate mixed Gaussian
Z = mixed_gaussian_2d(X, Y, comp)

# Print some statistics
print(f"Maximum value: {np.max(Z):.6f}")
print(f"Minimum value: {np.min(Z):.6f}")
print(f"Sum of all values: {np.sum(Z):.6f}")

# Reshape 2D to 1D function (row-major by default)
Zfunction_1d = Z.flatten()
print(f"1D shape (simple flatten): {Zfunction_1d.shape}")

# Verify round-trip conversion
Zfunction_2d_recovered = Zfunction_1d.reshape(Z.shape)
print(f"\nRound-trip conversion successful: {np.allclose(Z, Zfunction_2d_recovered)}")

# Plotting
fig = plt.figure(figsize=(5, 15))

# 2D contour plot
plt.subplot(3, 1, 1)
contour = plt.contour(X, Y, Z, levels=20)
plt.colorbar(contour)
plt.title('2D Mixed Gaussian - Contour')
plt.xlabel('X')
plt.ylabel('Y')

# 2D filled contour plot
plt.subplot(3, 1, 2)
plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar()
plt.title('2D Mixed Gaussian - Filled Contour')
plt.xlabel('X')
plt.ylabel('Y')

# 3D surface plot
ax = fig.add_subplot(3, 1, 3, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
plt.colorbar(surf)
ax.set_title('2D Mixed Gaussian - 3D Surface')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.tight_layout()
plt.savefig("mix_gaussian_1.png")

# Quantics tensor generation
qtensor_from_func1d = convert_1d_to_quantics_tensor(Zfunction_1d, x_dim + y_dim)
func1d_from_qtensor = convert_quantics_tensor_to_1d(qtensor_from_func1d)

print(f"Error - Function 1D -> quantics tensor -> Function 1D: {tl.norm(func1d_from_qtensor - Zfunction_1d)}")

# Save the data
filepath = "/home/zmeng5/QTTM/datasets/qtensor_gaussian/mix2d_gaussian_1.hdf5"
save_quantics_tensor_hdf5(qtensor_from_func1d, filepath)

qtensor_new, metadata = load_quantics_tensor_hdf5(filepath)
print(f"Difference between the real tensor and hdf5-loaded tensor: {tl.norm(qtensor_from_func1d - qtensor_new)}")