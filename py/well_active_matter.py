import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange

from the_well.benchmark.metrics import VRMSE
from the_well.data import WellDataset
from the_well.utils.download import well_download

# Reshape 256x256 field to quantics (2^8 x 2^8) format
def reshape_to_quantics(field_2d):
    # Each spatial point (i,j) maps to binary representation
    result = np.zeros((2,) * 16)
    
    for i in range(256):
        for j in range(256):
            # Convert i,j to binary representation
            i_bits = [(i >> k) & 1 for k in range(8)]
            j_bits = [(j >> k) & 1 for k in range(8)]
            
            # Store in quantics tensor
            result[tuple(i_bits + j_bits)] = field_2d[i, j]
    
    return result

# Convert Well data to QTT-ready format
def prepare_qtt_data(dataset_item):
    input_fields = dataset_item['input_fields']  # Q: Input or Output fields?
    
    # Reshape to quantics format
    # 256 = 2^8, so we get 8 binary dimensions per spatial axis
    qtt_shape = (2,) * 16 + (input_fields.shape[0], input_fields.shape[-1])
    qtt_data = np.zeros(qtt_shape)
    
    for t in range(input_fields.shape[0]):
        for f in range(input_fields.shape[-1]):
            field_2d = input_fields[t, :, :, f]
            # Convert to quantics representation
            qtt_data[:,:,:,:,:,:,:,:,:,:,:,:,:,:,:,:, t, f] = reshape_to_quantics(field_2d)
    
    tensor = qtt_data[:,:,:,:,:,:,:,:,:,:,:,:,0,0,0,0,0,0]

    return tensor

base_path = "./datasets"  

dataset = WellDataset(
    well_base_path=base_path,
    well_dataset_name="active_matter",
    well_split_name="train",
    n_steps_input=4,
    n_steps_output=1,
    use_normalization=False,
)

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