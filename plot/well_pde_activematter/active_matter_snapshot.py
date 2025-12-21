import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from the_well.data import WellDataset

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

print(f"Field names: {dataset.metadata.field_names}")

input_fields = item['input_fields']
print(f"Original input fields shape: {input_fields.shape}")

space_grid = item['space_grid']
print(f"Original space grid shape: {space_grid.shape}")

x_coords = space_grid[:, :, 0]
y_coords = space_grid[:, :, 1]

# Choose which time step and field to plot
time_step = 50

field_name = [r'Concentration $c$', r'Velocity $u_x$', r'Velocity $u_y$', r'Orientation $D_{xx}$', r'Orientation $D_{xy}$', r'Orientation $D_{yx}$', r'Orientation $D_{yy}$']

plot_field_no = 7
fig, axes = plt.subplots(1, plot_field_no, figsize=(14, 2))
for field_index in range(plot_field_no):
    function_2d = input_fields[time_step, :, :, field_index]
    im1 = axes[field_index].imshow(function_2d, cmap='viridis', aspect='auto',extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()])
    axes[field_index].set_title(field_name[field_index])
    axes[field_index].set_xticks([0,10], labelsize=10)
    axes[field_index].set_yticks([0,10], labelsize=10)
    
    

plt.tight_layout()
plt.savefig("active_matter_snapshot.png")
plt.savefig("active_matter_snapshot.pdf")

