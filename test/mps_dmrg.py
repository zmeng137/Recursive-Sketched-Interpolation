import os
import sys
import h5py
import numpy as np
import tensorly as tl

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'py'))

from multiply_rsi import HadamardTT_RSI   # RSI method for TT product
from multiply_direct import HadamardTT_direct
from tt_rounding import TT_rounding

def mps_all_contract(mps):
    result = np.array([[1.0]])

    for core in mps:
        result = result @ core

    return result[0, 0]

def mps_norm(mps):
    n = len(mps)
    mps_copy = mps.copy()
    
    # Contract each core with all-ones vector
    for i in range(n):
        # Sum over physical dimension: contract with all-ones
        mps_copy[i] = mps_copy[i][:,0,:] + mps_copy[i][:,1,:] + mps_copy[i][:,2,:]  #np.sum(mps_copy[i], axis=1)  # Shape: (r_i, r_{i+1})

    norm = mps_all_contract(mps_copy)

    return norm

def mps_diagonal(mps_square):
    n = len(mps_square)
    mps_copy = mps_square.copy()
    for i in range(n):
        mps_copy[i] = np.sum(mps_copy[i], axis=1)  # Shape: (r_i, r_{i+1})

    energy = 0.0
    for j in range(n-1):
        mps_copy_ = mps_copy.copy()
        mps_copy_[j] = mps_square[j][:,0,:] - mps_square[j][:,2,:]  
        mps_copy_[j+1] = mps_square[j+1][:,0,:] - mps_square[j+1][:,2,:]
    
        contrib = mps_all_contract(mps_copy_)
        energy = energy + contrib

    return energy


def readh5_mps(filePath):   
    with h5py.File(filePath, "r") as f:
        num_sites = f["num_sites"][()]
        energy = f["energy"][()]
        energy_diag = f["energy_diag"][()]
        
        mps = []
        for i in range(1, num_sites + 1):
            tensor = f[f"tensor_{i}"][:]
            shape = f[f"shape_{i}"][:]
            mps.append(tensor)
            
            if i == 1:
                print(f"Site {i} (first): shape {tensor.shape} = (physical={shape[0]}, right_bond={shape[1]})")
            elif i == num_sites:
                print(f"Site {i} (last): shape {tensor.shape} = (left_bond={shape[0]}, physical={shape[1]})")
            else:
                print(f"Site {i}: shape {tensor.shape} = (left_bond={shape[0]}, physical={shape[1]}, right_bond={shape[2]})")
        
        print(f"\nGround state energy: {energy}")

    mps.reverse()
    mps[0] = mps[0].reshape(1,mps[0].shape[0],mps[0].shape[1])
    mps[-1] = mps[-1].reshape(mps[-1].shape[0],mps[-1].shape[1],1)

    mps_rank = [mps[0].shape[0]] + [mps[k].shape[2] for k in range(len(mps)-1)] + [mps[-1].shape[2]]
    print(f"\nMPS rank: {mps_rank}\n")

    return mps, num_sites, energy, energy_diag

# Load MPS from file
filePath = "/home/zmeng5/QTTM/datasets/itensor_dmrg_mps/n15_system/psi_maxdim10_n15.h5"
mps, num_sites, energy, energy_diag = readh5_mps(filePath)

print("\n == Evaluation of True func == \n")

# Measurement: Norm bias: |1-norm|
Normbias_dict_rsi = {}
Normbias_dict_dir = {}

# Measurement: Diagonal energy deviation: |E_true - E_approx|
Energydiag_dict_rsi = {}
Energydiag_dict_dir = {}

# Measurement: Relative error
rel_error_dict_rsi = {}
rel_error_dict_dir = {}

fulleval_thres = 20
if num_sites <= fulleval_thres:
    full_tensor = tl.tt_to_tensor(mps)
    full_tensor_diag = full_tensor * full_tensor

r_max = [50,100,150] 
contract_number = [2] 
oversampling = 10

print("\n ====== RSI ====== \n")
for co in contract_number:
    for rm in r_max:
        seed = 1
        eps=0
        skdim = int(rm/2) + oversampling
        TTg_rsi, TTRank_rsi, _  = HadamardTT_RSI(mps, mps, co, rm, eps, skdim, seed)
        
        print("\n == Evaluation of Approx == \n")
        norm_rsi = mps_norm(TTg_rsi)    
        energydiag_rsi = mps_diagonal(TTg_rsi)

        Energydiag_dict_rsi[f"cont_no={co}; rmax={rm}; skdim={skdim}"] = np.abs(energy_diag - energydiag_rsi)
        Normbias_dict_rsi[f"cont_no={co}; rmax={rm}; skdim={skdim}"] = np.abs(1 - norm_rsi)
        if num_sites <= fulleval_thres:
            g_rsi = tl.tt_to_tensor(TTg_rsi)
            rel_error = np.linalg.norm(full_tensor_diag - g_rsi) / np.linalg.norm(full_tensor_diag)
            rel_error_dict_rsi[f"cont_no={co}; rmax={rm}; skdim={skdim}"] = rel_error

# Output errors
print("\n === RSI Method Diag Energy deviation ===")
for key, value in Energydiag_dict_rsi.items():
    print(f"{key}: {value}")
print("\n =========================\n")

print("\n === RSI Method Relative Error ===")
for key, value in rel_error_dict_rsi.items():
    print(f"{key}: {value}")
print("\n =========================\n")

# Direct method
# direct kronecker
print("\n ====== Direct method (Kronecker product) ====== \n")
TTg_direct = HadamardTT_direct(mps, mps)
TTrank_direct = [TTg_direct[0].shape[0]] + [TTg_direct[k].shape[2] for k in range(len(TTg_direct)-1)] + [TTg_direct[-1].shape[2]]

# Rounding recompression
print("\n == Direct method (TT-rounding) == \n")
for rm in r_max:
    TTg_direct_trunc = TT_rounding(TTg_direct, 0, rm)   # Post re-compression
    TTrank_direct_trunc = [TTg_direct_trunc[0].shape[0]] + [TTg_direct_trunc[k].shape[2] for k in range(len(TTg_direct_trunc)-1)] + [TTg_direct_trunc[-1].shape[2]]

    print("\n == Evaluation of Approx == \n")
    norm_dir = mps_norm(TTg_direct_trunc) 
    energydiag_dir = mps_diagonal(TTg_direct_trunc)  
    Normbias_dict_dir[f"direct rounding rmax={rm}"] = np.abs(1 - norm_dir)
    Energydiag_dict_dir[f"direct rounding rmax={rm}"] = np.abs(energy_diag - energydiag_dir)
    if num_sites <= fulleval_thres:
        g_direct = tl.tt_to_tensor(TTg_direct_trunc)
        rel_error_dir = np.linalg.norm(full_tensor_diag - g_direct) / np.linalg.norm(full_tensor_diag)
        rel_error_dict_dir[f"direct rounding rmax={rm}"] = rel_error_dir

print("\n === DIR Method Diag Energy deviation ===\n")
for key, value in Energydiag_dict_dir.items():
    print(f"{key}: {value}")
print("\n =========================\n")

print("\n === DIR Method Relative Error ===\n")
for key, value in rel_error_dict_dir.items():
    print(f"{key}: {value}")
print("\n =========================\n")
