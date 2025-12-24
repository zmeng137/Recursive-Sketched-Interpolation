import h5py
import numpy as np
import tensorly as tl
import time as tm
from multiply_rsi import HadamardTT_RSI   # RSI method for TT product
from multiply_direct import HadamardTT_direct
from tt_rounding import TT_rounding
from error_eval import compute_tt_relative_error

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

    return mps, energy, energy_diag

filePath = "/home/zmeng5/QTTM/datasets/quantum_dmrg/psi_maxdim50_n50.h5"
mps, energy, energy_diag = readh5_mps(filePath)

print("\n == Evaluation of True func == \n")

Zdeviation_dict_rsi = {}
Zdeviation_dict_dir = {}
Energydiag_dict_rsi = {}
Energydiag_dict_dir = {}

r_max = [100, 200, 300]
contract_number = [2] 
for co in contract_number:
    for rm in r_max:
        print("\n == RSI == \n")
        seed = 0
        eps=0
        skdim = int(rm/2)+20
        TTg_rsi, TTRank_rsi, _  = HadamardTT_RSI(mps, mps, co, rm, eps, skdim, seed)
        
        print("\n == Evaluation of Approx == \n")
        norm_rsi = mps_norm(TTg_rsi)    
        energydiag_rsi = mps_diagonal(TTg_rsi)

        Energydiag_dict_rsi[f"cont_no={co}; rmax={rm}; skdim={skdim}"] = np.abs(energy_diag - energydiag_rsi)
        Zdeviation_dict_rsi[f"cont_no={co}; rmax={rm}; skdim={skdim}"] = np.abs(1 - norm_rsi)

# Output errors
print("\n === RSI Method Diag Energy deviation ===\n")
for key, value in Energydiag_dict_rsi.items():
    print(f"{key}: {value}")
print("\n =========================\n")

print("\n === RSI Method Z deviation ===\n")
for key, value in Zdeviation_dict_rsi.items():
    print(f"{key}: {value}")
print("\n =========================\n")


# Direct method
# direct kronecker
print("\n == direct == \n")
TTg_direct = HadamardTT_direct(mps, mps)
TTrank_direct = [TTg_direct[0].shape[0]] + [TTg_direct[k].shape[2] for k in range(len(TTg_direct)-1)] + [TTg_direct[-1].shape[2]]

# Rounding recompression
for rm in r_max:
    print("\n == TT rounding == \n")
    TTg_direct_trunc = TT_rounding(TTg_direct, 0, rm)   # Post re-compression
    TTrank_direct_trunc = [TTg_direct_trunc[0].shape[0]] + [TTg_direct_trunc[k].shape[2] for k in range(len(TTg_direct_trunc)-1)] + [TTg_direct_trunc[-1].shape[2]]

    print("\n == Evaluation of Approx == \n")
    norm_dir = mps_norm(TTg_direct_trunc) 
    energydiag_dir = mps_diagonal(TTg_direct_trunc)  
    Zdeviation_dict_dir[f"direct rounding rmax={rm}"] = np.abs(1 - norm_dir)
    Energydiag_dict_dir[f"direct rounding rmax={rm}"] = np.abs(energy_diag - energydiag_dir)

print("\n === DIR Method Diag Energy deviation ===\n")
for key, value in Energydiag_dict_dir.items():
    print(f"{key}: {value}")
print("\n =========================\n")


print("\n === DIR Method Z deviation ===\n")
for key, value in Zdeviation_dict_dir.items():
    print(f"{key}: {value}")
print("\n =========================\n")



#mps_square_rounding = TT_rounding(mps_square_direct, 0, rmax)
#approx_rsi = tl.tt_to_tensor(mps_square_rsi)
#approx_dir = tl.tt_to_tensor(mps_square_rounding)
#relerr_rsi = np.linalg.norm(real_g - approx_rsi) / np.linalg.norm(real_g)
#relerr_dir = np.linalg.norm(real_g - approx_dir)

#start = tm.time()
#error = compute_tt_relative_error(mps_square_direct, mps_square_rsi)
#norm_direct = mps_contract_ones(mps_square_direct)
#elapsed = tm.time() - start

#print(f"RSI Relative error: {relerr_rsi}")
#print(f"Norm direct: {norm_direct:.6e}")
#print(f"Norm RSI: {norm_rsi:.6e}")
#print(f"Rel error of Norm RSI: {1-norm_rsi:.6e}")
#print(f"Computation time: {elapsed*1000:.2f} ms")
#print(f"Direct Relative error: {relerr_dir}")



pass