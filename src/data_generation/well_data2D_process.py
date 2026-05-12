import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from the_well.data import WellDataset
from utils import save_quantics_tensor_hdf5, load_quantics_tensor_hdf5, convert_1d_to_quantics_tensor, convert_quantics_tensor_to_1d

base_path = "/mnt/CROSS/Well/"  

dataset = WellDataset(
    well_base_path=base_path,
    well_dataset_name="active_matter",
    well_split_name="train",
    n_steps_input=80,
    n_steps_output=1,
    use_normalization=False,
)

item = dataset[0]

# Extract input fields (n_steps_input, 128, 384, n_fields)
input_fields = item['input_fields']
print(f"Original input fields shape: {input_fields.shape}")

# SPATIAL TRUNCATION: Reduce from (128, 384) to (128, 256)
# Option 1: Take first 256 columns
#input_fields_truncated = input_fields[:, :, :256, :]

# Option 2: Take center 256 columns (recommended for symmetry)
# start_col = (384 - 256) // 2  # = 64
# input_fields_truncated = input_fields[:, :, start_col:start_col+256, :]

# Option 3: Take last 256 columns
# input_fields_truncated = input_fields[:, :, -256:, :]

#print(f"Truncated input fields shape: {input_fields_truncated.shape}")

# Also truncate the spatial grid
space_grid = item['space_grid']
print(f"Original space grid shape: {space_grid.shape}")

# Apply same truncation to spatial grid
#space_grid_truncated = space_grid[:, :256, :]  # or use center/end based on above choice
#print(f"Truncated space grid shape: {space_grid_truncated.shape}")

x_coords = space_grid[:, :, 0]
y_coords = space_grid[:, :, 1]

# Choose which time step and field to plot
time_step = 50
field_index = 6

# Extract the 2D function at specified time step (now 128×256)
function_2d = input_fields[time_step, :, :, field_index]
print(f"\nFunction 2D shape: {function_2d.shape}")

# Statistics
print(f"\nFunction statistics:")
print(f"Min value: {function_2d.min():.4f}")
print(f"Max value: {function_2d.max():.4f}")
print(f"Mean value: {function_2d.mean():.4f}")
print(f"Standard deviation: {function_2d.std():.4f}")

# Reshape 2D to 1D function (row-major)
function_1d_simple = function_2d.flatten()
print(f"1D shape: {function_1d_simple.shape}")  # Should be (32768,) = 128*256

# Verify round-trip conversion
function_2d_recovered = function_1d_simple.reshape(function_2d.shape)
print(f"\nRound-trip conversion successful: {np.allclose(function_2d, function_2d_recovered)}")

# Visualization
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

# Original 2D function (truncated)
im1 = axes[0].imshow(function_2d, cmap='viridis', aspect='auto')
axes[0].set_title(f'Truncated 2D Function ({function_2d.shape[0]}×{function_2d.shape[1]})')
axes[0].set_xlabel('Y index')
axes[0].set_ylabel('X index')
plt.colorbar(im1, ax=axes[0])

# 1D representation
axes[1].plot(function_1d_simple[:1000])
axes[1].set_title('1D Representation (first 1000 elements)')
axes[1].set_xlabel('1D Index')
axes[1].set_ylabel('Function Value')

# Recovered 2D function
im3 = axes[2].imshow(function_2d_recovered, cmap='viridis', aspect='auto')
axes[2].set_title('Recovered 2D Function')
axes[2].set_xlabel('Y index')
axes[2].set_ylabel('X index')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig("active_matter_test.png")

# Quantics tensor generation
# For 128×256 = 32768 = 2^15, you need 15 qubits
qtensor_from_func1d = convert_1d_to_quantics_tensor(function_1d_simple, 16)
func1d_from_qtensor = convert_quantics_tensor_to_1d(qtensor_from_func1d)
func1d_from_qtensor = torch.from_numpy(func1d_from_qtensor)
print(f"Error - Function 1D -> quantics tensor -> Function 1D: {torch.norm(func1d_from_qtensor - function_1d_simple)}")

# Save the data
filepath = "/home/zmeng5/QTTM/datasets/qtensor_well/active_matter_t50_Dyy.hdf5"
save_quantics_tensor_hdf5(qtensor_from_func1d, filepath)

qtensor_new, metadata = load_quantics_tensor_hdf5(filepath)
print(f"Difference between the real tensor and hdf5-loaded tensor: {tl.norm(qtensor_from_func1d - qtensor_new)}")