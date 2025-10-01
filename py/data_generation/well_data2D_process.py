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
    n_steps_input=10,
    n_steps_output=1,
    use_normalization=False,
)

item = dataset[0]  # Which dataset

# The data structure
print("Available keys:", list(item.keys()))
print("Input fields shape:", item['input_fields'].shape)
print("Output fields shape:", item['output_fields'].shape)

# Extract input fields (n_steps_input, 256, 256, n_fields)
input_fields = item['input_fields']
print(f"Input fields shape: {input_fields.shape}")

# Spatial grid
space_grid = item['space_grid']
print(f"Space grid shape: {space_grid.shape}")

# Extract x and y coordinates
x_coords = space_grid[:, :, 0]  # x coordinates
y_coords = space_grid[:, :, 1]  # y coordinates

# Choose which time step and field to plot
time_step = 7    # Which time step (0 to n_steps_input-1)
field_index = 1  # Which field (change this to plot different fields)

# Extract the 2D function at specified time step
function_2d = input_fields[time_step, :, :, field_index]

# Statistics about the function
print(f"\nFunction statistics:")
print(f"Min value: {function_2d.min():.4f}")
print(f"Max value: {function_2d.max():.4f}")
print(f"Mean value: {function_2d.mean():.4f}")
print(f"Standard deviation: {function_2d.std():.4f}")

# Reshape 2D to 1D function (row-major by default)
function_1d_simple = function_2d.flatten()
print(f"1D shape (simple flatten): {function_1d_simple.shape}")

# Manual indexing to verify the ordering. Verify methods give the same result
#function_1d_manual = np.zeros(256 * 256)
#idx = 0
#for i in range(256):  # x index
#    for j in range(256):  # y index
#        function_1d_manual[idx] = function_2d[i, j]
#        idx += 1
#print(f"1D shape (manual): {function_1d_manual.shape}")
#print(f"Methods are equivalent: {np.allclose(function_1d_simple, function_1d_manual)}")

# Verify round-trip conversion
function_2d_recovered = function_1d_simple.reshape(function_2d.shape)
print(f"\nRound-trip conversion successful: {np.allclose(function_2d, function_2d_recovered)}")

# Visualization to understand the reshaping
fig, axes = plt.subplots(3, 1, figsize=(8, 15))

# Original 2D function
im1 = axes[0].imshow(function_2d, cmap='viridis')
axes[0].set_title('Original 2D Function')
axes[0].set_xlabel('Y index')
axes[0].set_ylabel('X index')
plt.colorbar(im1, ax=axes[0])

# 1D representation (show first 1000 elements)
axes[1].plot(function_1d_simple[:1000])
axes[1].set_title('1D Representation (first 1000 elements)')
axes[1].set_xlabel('1D Index')
axes[1].set_ylabel('Function Value')

# Recovered 2D function
im3 = axes[2].imshow(function_2d_recovered, cmap='viridis')
axes[2].set_title('Recovered 2D Function')
axes[2].set_xlabel('Y index')
axes[2].set_ylabel('X index')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig("active_matter_test.png")

# Quantics tensor generation
qtensor_from_func1d = convert_1d_to_quantics_tensor(function_1d_simple, 16)
func1d_from_qtensor = convert_quantics_tensor_to_1d(qtensor_from_func1d)
func1d_from_qtensor = torch.from_numpy(func1d_from_qtensor)
print(f"Error - Function 1D -> quantics tensor -> Function 1D: {torch.norm(func1d_from_qtensor - function_1d_simple)}")


# Save the data
filepath = "/home/zmeng5/QTTM/datasets/qtensor_well/qt_active_matter_1.hdf5"
save_quantics_tensor_hdf5(qtensor_from_func1d, filepath)

qtensor_new, metadata = load_quantics_tensor_hdf5(filepath)
print(f"Difference between the real tensor and hdf5-loaded tensor: {tl.norm(qtensor_from_func1d - qtensor_new)}")