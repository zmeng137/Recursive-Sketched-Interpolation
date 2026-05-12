import os
import sys
import torch
import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from the_well.data import WellDataset
from utils import save_quantics_tensor_hdf5, load_quantics_tensor_hdf5, convert_1d_to_quantics_tensor, convert_quantics_tensor_to_1d

base_path = "/mnt/CROSS/Well/"  

dataset = WellDataset(
    well_base_path=base_path,
    well_dataset_name="MHD_256",
    well_split_name="test",
    n_steps_input=10,
    n_steps_output=1,
    use_normalization=False,
)

item = dataset[0]  # Which dataset

# The data structure
print("Available keys:", list(item.keys()))

# Extract input/output fields
input_fields = item['input_fields']
output_fields = item['output_fields']
print(f"Input fields shape: {input_fields.shape}")
print(f"Output fields shape: {output_fields.shape}")

# Spatial grid
space_grid = item['space_grid']
print(f"Space grid shape: {space_grid.shape}")

# Extract x and y coordinates
x_coords = space_grid[:, :, :, 0]  # x coordinates
y_coords = space_grid[:, :, :, 1]  # y coordinates
z_coords = space_grid[:, :, :, 2]  # z coordinates

# Choose which time step and field to plot
time_step = 5    # Which time step (0 to n_steps_input-1)
field_index = 5  # Which field (change this to plot different fields)

# Extract the 3D function at specified time step
function_3d = input_fields[time_step, :, :, :, field_index]

# Statistics about the function
print(f"\nFunction statistics:")
print(f"Min value: {function_3d.min():.4f}")
print(f"Max value: {function_3d.max():.4f}")
print(f"Mean value: {function_3d.mean():.4f}")
print(f"Standard deviation: {function_3d.std():.4f}")

# Reshape 2D to 1D function (row-major by default)
function_1d_simple = function_3d.flatten()
print(f"1D shape (simple flatten): {function_1d_simple.shape}")

# Verify round-trip conversion
function_3d_recovered = function_1d_simple.reshape(function_3d.shape)
print(f"\nRound-trip conversion successful: {np.allclose(function_3d, function_3d_recovered)}")

def plot_isosurface(x, y, z, values, isovalue=None):
    """Create isosurface plot using Plotly"""
    if isovalue is None:
        isovalue = np.mean(values)
    
    fig = go.Figure(data=go.Isosurface(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=values.flatten(),
        isomin=np.min(values),
        isomax=np.max(values),
        surface_count=3,  # Number of isosurfaces
        colorscale='Viridis',
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))
    
    fig.update_layout(
        title=f'Isosurface Visualization (Isovalue: {isovalue:.3f})',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=800,
        height=600,
    )
    
    fig.write_image("TheWell_3D.png")

def plot_1d_vectorization(func1d):
    """Plot the 1D function"""
    plt.figure()
    plt.plot(func1d[:1000])
    plt.xlabel("1D Index")
    plt.ylabel("Function Value")
    plt.title("1D Representation (first 1000 elements)")
    plt.savefig("Vectorization_1DFunc.png")
    
#plot_isosurface(x_coords.numpy(), y_coords.numpy(), z_coords.numpy(), function_3d.numpy())
#plot_1d_vectorization(function_1d_simple)

# Quantics tensor generation
qtensor_from_func1d = convert_1d_to_quantics_tensor(function_1d_simple, 24)
func1d_from_qtensor = convert_quantics_tensor_to_1d(qtensor_from_func1d)
func1d_from_qtensor = torch.from_numpy(func1d_from_qtensor)
print(f"Error - Function 1D -> quantics tensor -> Function 1D: {torch.norm(func1d_from_qtensor - function_1d_simple)}")

# Save the data
filepath = "/home/zmeng5/QTTM/datasets/qtensor_well/mhd256_vy.hdf5"
save_quantics_tensor_hdf5(qtensor_from_func1d, filepath)

qtensor_new, metadata = load_quantics_tensor_hdf5(filepath)
print(f"Difference between the real tensor and hdf5-loaded tensor: {tl.norm(qtensor_from_func1d - qtensor_new)}")