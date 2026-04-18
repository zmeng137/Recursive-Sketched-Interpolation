import h5py
import numpy as np

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