import sys
import os
import h5py

import pandas as pd

from multiply_direct import HadamardTT_direct
from multiply_rsi import HadamardTT_RSI


sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MLA-Toolkit', 'py'))

file1 = '/home/zmeng5/QTTM/datasets/quantum_chem/slater_XXX_(1,0,0)_dd20_rk150.h5'
file2 = "/home/zmeng5/QTTM/datasets/quantum_chem/slater_XXX_shift14013_(1,0,0)_dd20_rk150.h5"
file3 = "/home/zmeng5/QTTM/datasets/quantum_chem/slater_XXX_shift14013_(2,1,-1)_dd20_rk150.h5"

def explore_h5(file_path):
    with h5py.File(file_path, 'r') as f:
        def print_structure(name, obj):
            print(name)
        f.visititems(print_structure)

def load_mps(filename):
    with h5py.File(filename, 'r') as f:
        # Get number of tensors
        num_tensors = len(f.keys())
        
        # Load into list
        mps_tensors = []
        for i in range(num_tensors):
            tensor = f[f'array_{i}'][:]
            mps_tensors.append(tensor)
            print(f"Tensor {i}: shape {tensor.shape}")
        
    return mps_tensors

print("Load TT_f1 ...")
TT_f1 = load_mps(file3)
print("\nLoad TT_f2 ...")
TT_f2 = load_mps(file2)

TT_f1_rank = [1] + [TT_f1[i].shape[2] for i in range(len(TT_f1))]
TT_f2_rank = [1] + [TT_f2[i].shape[2] for i in range(len(TT_f2))]
print(f"\nTT_f1 Bond dimension {TT_f1_rank}, max {max(TT_f1_rank)}")
print(f"TT_f2 Bond dimension {TT_f2_rank}, max {max(TT_f2_rank)}")

print("\n Hadamard product of f1 and f2 ...")
r_g_max = 50
eps = 1e-10
sketch_dim = 50
TT_g1, TT_g1_rank, _ = HadamardTT_RSI(TT_f1, TT_f2, 2, r_g_max, eps, sketch_dim, 0)
print(f"\nTT_g1 Bond dimension {TT_g1_rank}, max {max(TT_g1_rank)}")

TT_g2 = HadamardTT_direct(TT_f1, TT_f2)
TT_g2_rank = [TT_g2[0].shape[0]] + [TT_g2[k].shape[2] for k in range(len(TT_g2)-1)] + [TT_g2[-1].shape[2]]
print(f"\nTT_g2 Bond dimension {TT_g2_rank}, max {max(TT_g2_rank)}")

# But how to check the accuracy?
# Idea: one by one query?