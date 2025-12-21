import h5py
import numpy as np
import tensorly as tl
import time as tm
from multiply_rsi import HadamardTT_RSI   # RSI method for TT product
from multiply_direct import HadamardTT_direct
from tt_rounding import TT_rounding
from error_eval import compute_tt_relative_error

def mps_contract_ones(cores):
    # Start with identity for the leftmost boundary
    result = np.array([[1.0]])
    
    # Contract each core with all-ones vector
    for core in cores:
        # Sum over physical dimension: contract with all-ones
        contracted = np.sum(core, axis=1)  # Shape: (r_i, r_{i+1})
        # Contract with accumulated result
        result = result @ contracted
    
    return result[0, 0]

def readh5_mps(filePath):   
    with h5py.File(filePath, "r") as f:
        num_sites = f["num_sites"][()]
        energy = f["energy"][()]
        
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

    return mps, energy

filePath = "/home/zmeng5/QTTM/datasets/quantum_dmrg/psi_3.h5"
mps, energy = readh5_mps(filePath)

#mps_square_direct = HadamardTT_direct(mps, mps)

print("\n == Evaluation of True func == \n")
real_f = tl.tt_to_tensor(mps)
real_g = real_f * real_f

relerror_dict_rsi = {}
Zdeviation_dict_rsi = {}
r_max = [50]
contract_number = [2] 
for co in contract_number:
    for rm in r_max:
        print("\n == RSI == \n")
        seed = 0
        eps=0
        skdim = rm
        TTg_rsi, TTRank_rsi, _  = HadamardTT_RSI(mps, mps, co, rm, eps, skdim, seed)
        
        print("\n == Evaluation of Approx == \n")
        g_rsi = tl.tt_to_tensor(TTg_rsi)
        err_g_rsi = np.linalg.norm(real_g - g_rsi) / np.linalg.norm(real_g)
        norm_rsi = mps_contract_ones(TTg_rsi)    
        relerror_dict_rsi[f"cont_no={co}; rmax={rm}; skdim={skdim}"] = err_g_rsi
        Zdeviation_dict_rsi[f"cont_no={co}; rmax={rm}; skdim={skdim}"] = 1 - norm_rsi

# Output errors
print("\n === RSI Method Errors ===\n")
for key, value in relerror_dict_rsi.items():
    print(f"{key}: {value}")
print("\n =========================\n")

print("\n === RSI Method Z deviation ===\n")
for key, value in Zdeviation_dict_rsi.items():
    print(f"{key}: {value}")
print("\n =========================\n")

'''
inner(psi', H, psi)
Hzz = \sum Sz_j Sz_j+1
inner(psi', Hzz, psi)
'''

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