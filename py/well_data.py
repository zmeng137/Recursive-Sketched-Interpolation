import numpy as np
import torch
import matplotlib.pyplot as plt

from the_well.data import WellDataset
from utils import save_quantics_tensor_hdf5

base_path = "./datasets"  

dataset = WellDataset(
    well_base_path=base_path,
    well_dataset_name="active_matter",
    well_split_name="train",
    n_steps_input=4,
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
time_step = 3    # Which time step (0 to n_steps_input-1)
field_index = 2  # Which field (change this to plot different fields)

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

# Convert 1D function to quantics tensor format recursively
def convert_1d_to_quantics_tensor(function, n_bits):
    if len(function) != 2**n_bits:
        raise ValueError(f"Function length {len(function)} != 2^{n_bits} = {2**n_bits}")
    
    if n_bits == 1:
        #quantics_tensor = torch.zeros(2,dtype=function.dtype)
        quantics_tensor = np.zeros(2,dtype=np.float32)
        quantics_tensor[0] = function[0]
        quantics_tensor[1] = function[1]
        return quantics_tensor
    
    # Create the quantics tensor
    quantics_shape = (2,) * n_bits
    #quantics_tensor = torch.zeros(quantics_shape, dtype=function.dtype)
    quantics_tensor = np.zeros(quantics_shape, dtype=np.float32)
    
    new_bit = n_bits - 1
    func_half_1 = function[0 : 2 ** new_bit]
    func_half_2 = function[2 ** new_bit :] 

    quantics_tensor[0,:] = convert_1d_to_quantics_tensor(func_half_1, new_bit)
    quantics_tensor[1,:] = convert_1d_to_quantics_tensor(func_half_2, new_bit)
    
    return quantics_tensor

# Quantics tensor generation
qtensor = convert_1d_to_quantics_tensor(function_1d_simple, 16)

# Save the data
filepath = "/home/zmeng5/QTTM/datasets/qtensor_well/qt_active_matter_0.hdf5"
save_quantics_tensor_hdf5(qtensor, filepath)



'''
item = dataset[0]
tensor = prepare_qtt_data(item)
pass


from tt_svd import TT_SVD
import tensorly as tl

#r_max = 12
#TT_cores = TT_SVD(test_tensor, r_max, 1e-8, 0)
#recon_f1 = tl.tt_to_tensor(TT_cores)
#error = tl.norm(test_tensor - recon_f1) / tl.norm(test_tensor)
#print(f"Relative error of f1 QTT at r_max = {r_max}: {error}")


from tensor_cross import TCI_2site, nested_initIJ_gen_rank1, TT_IDPRRLDU

r_max = 12
TT_cores = TT_IDPRRLDU(tensor, r_max, 1e-8, 0)
recon_f1 = tl.tt_to_tensor(TT_cores)
error = tl.norm(tensor - recon_f1) / tl.norm(tensor)
print(f"Relative error of f1 QTT at r_max = {r_max}: {error}")


dim = len(tensor.shape)
Nested_I_rank1, Nested_J_rank1 = nested_initIJ_gen_rank1(dim)

# PROBLEM FOR TCI! -> no convergence for the well data?

# TCI-2site of f1
eps = 1e-8
r_max = 12
TT_cross_f1, TT_cores_f1, TTRank_f1, pr_set_f1, pc_set_f1, interp_I_f1, interp_J_f1 = TCI_2site(tensor, eps, r_max, Nested_I_rank1, Nested_J_rank1)
recon_f1 = tl.tt_to_tensor(TT_cores_f1)
error = tl.norm(tensor - recon_f1) / tl.norm(tensor)
print(f"Relative error of f1 QTT at r_max = {r_max}: {error}")
'''