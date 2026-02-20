import os
import sys
import h5py
import math as ma
import numpy as np
import time as tm
import tensorly as tl
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'py'))
from utils import load_quantics_tensor_hdf5, convert_quantics_tensor_to_1d, size_tt, load_quantics_tensor_formula, tt_to_tensor_tensordot
from tci import TT_IDPRRLDU_L2R                                      # One-side/sweep TCI (TT-ID)
from multiply_rsi import HadamardTT_RSI, HadamardTT_RSI_fs           # RSI method for TT product
from multiply_direct import HadamardTT_direct, HadamardTT_direct_fs  # Direct method for TT product
from tt_rounding import TT_rounding, TT_rounding_ID                  # TT rounding for rank compression

''' ===================== Tensor Generation ===================== '''

def quantics_function_tensor():
    # Load quantics function tensors from synthetic formulas 
    qtensor_f1, _ = load_quantics_tensor_formula(5, 20)
    qtensor_f2, _ = load_quantics_tensor_formula(6, 20)
 
    # Load quantics function tensors from hdf5 files 
    #filePath_f1 = "/home/zmeng5/QTTM/datasets/qtensor_well/active_matter_t50_Dxy.hdf5"
    #filePath_f2 = "/home/zmeng5/QTTM/datasets/qtensor_well/active_matter_t50_Dxy.hdf5"
    #qtensor_f1, _ = load_quantics_tensor_hdf5(filePath_f1)
    #qtensor_f2, _ = load_quantics_tensor_hdf5(filePath_f2)

    # TT Decomposition of f1 and f2
    r_max_f1 = 10
    r_max_f2 = 10
    eps = 0

    TTCore_f1, TTRank_f1, _ = TT_IDPRRLDU_L2R(qtensor_f1, r_max_f1, eps, 0)
    TTCore_f2, TTRank_f2, _ = TT_IDPRRLDU_L2R(qtensor_f2, r_max_f2, eps, 0)
    #TTCore_f1 = tl.decomposition.tensor_train(qtensor_f1, r_max_f1)
    #TTCore_f2 = tl.decomposition.tensor_train(qtensor_f2, r_max_f2)
    #TTRank_f1 = [TTCore_f1[i].shape[2] for i in range(len(TTCore_f1)-1)]
    #TTRank_f2 = [TTCore_f2[i].shape[2] for i in range(len(TTCore_f2)-1)]
    
    recon_f1 = tl.tt_to_tensor(TTCore_f1)
    recon_f2 = tl.tt_to_tensor(TTCore_f2)
    error_f1 = tl.norm(qtensor_f1 - recon_f1) / tl.norm(qtensor_f1)
    error_f2 = tl.norm(qtensor_f2 - recon_f2) / tl.norm(qtensor_f2)
    print(f"Relative error of TT_f1 (r_max = {np.max(TTRank_f1)}): {error_f1}")
    print(f"Relative error of TT_f2 (r_max = {np.max(TTRank_f2)}): {error_f2}")
    print(f"Size of full qtensor_f {qtensor_f1.size}. Size of f1 QTT {size_tt(TTCore_f1)}, QTT compression ratio {qtensor_f1.size / size_tt(TTCore_f1)}")
    print(f"Size of full qtensor_f {qtensor_f2.size}. Size of f1 QTT {size_tt(TTCore_f2)}, QTT compression ratio {qtensor_f2.size / size_tt(TTCore_f2)}")
    

    ifPlot = False
    if ifPlot:
        f1_1d = convert_quantics_tensor_to_1d(qtensor_f1)
        f2_1d = convert_quantics_tensor_to_1d(qtensor_f2)
        ngrid = len(f1_1d)
        xgrid = np.linspace(0, 1, ngrid)
        plt.figure(figsize=(10,4))
        plt.plot(xgrid, f1_1d, label='f1(x)', linewidth=1)
        plt.plot(xgrid, f2_1d, label='f2(x)', linewidth=1)
        plt.xlabel('x', fontsize=14)
        plt.ylabel('Function value', fontsize=14)
        plt.title('1D Functions from Quantics Tensors', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid()
        plt.savefig('gaussian_1d_plot.png', dpi=300)

    return TTCore_f1, TTCore_f2, qtensor_f1, qtensor_f2

tt_f1, tt_f2, qf1, qf2 = quantics_function_tensor()

''' ===================== TT Multiplication ===================== '''
print("\n === Hadamard Product Test ===\n")

# tensors to be multiplied
q_fs = {}
q_fs[0] = qf1
q_fs[1] = qf2
q_fs[2] = qf2
q_fs[3] = qf1
#q_fs[4] = qf2


# TTs to be multiplied
tt_fs = {}
tt_fs[0] = tt_f1
tt_fs[1] = tt_f2
tt_fs[2] = tt_f2
tt_fs[3] = tt_f1
#tt_fs[4] = tt_f2


# Real product 
real_g = q_fs[0]
for i in range(1, len(q_fs)):
    real_g = real_g * q_fs[i]

# Error storage
ifEval = True
error_dict_rsi = {}
error_dict_rsi_round = {}
error_dict_direct = {}

# Recursive Sketching Interpolation (RSI) Algorithm
contract_number = [2]
r_max = [4,6,8,10,12]
oversampling = [5]

for con in contract_number:
    for rm in r_max:
        for p in oversampling:
            seed = 0
            eps=0
            sketch_dim = int(rm/2) + p # oversampling = ...
            TTg_rsi, TTRank_rsi, _  = HadamardTT_RSI_fs(tt_fs, con, rm, eps, sketch_dim, seed)
            
            if ifEval:
                g_rsi = tl.tt_to_tensor(TTg_rsi)
                err_g_rsi = np.linalg.norm(real_g - g_rsi) / np.linalg.norm(real_g)
                error_dict_rsi[f"cont_no={con}; rmax={rm}; oversampling={p}"] = err_g_rsi

#rsi_rounding_rank = [10,15,20,25]
#for rm in rsi_rounding_rank:
#    TTg_rsi_round = TT_rounding(TTg_rsi, 0, rm)
#    g_rsi_round = tl.tt_to_tensor(TTg_rsi_round)
#    err_g_rsi_round = np.linalg.norm(real_g - g_rsi_round) / np.linalg.norm(real_g)
#    error_dict_rsi_round[f"cont_no={2}; rmax={rm}; oversampling={p}"] = err_g_rsi_round

# Output errors
print("\n === RSI Method Errors ===\n")
for key, value in error_dict_rsi.items():
    print(f"{key}: {value}")
print("\n =========================\n")

print("\n === RSI Method (rounding) Errors ===\n")
for key, value in error_dict_rsi_round.items():
    print(f"{key}: {value}")
print("\n =========================\n")

# Direct method
# direct kronecker
TTg_direct = HadamardTT_direct_fs(tt_fs)
TTrank_direct = [TTg_direct[0].shape[0]] + [TTg_direct[k].shape[2] for k in range(len(TTg_direct)-1)] + [TTg_direct[-1].shape[2]]

# Rounding recompression
for rm in r_max:
    TTg_direct_trunc = TT_rounding(TTg_direct, 0, rm)   # Post re-compression
    TTrank_direct_trunc = [TTg_direct_trunc[0].shape[0]] + [TTg_direct_trunc[k].shape[2] for k in range(len(TTg_direct_trunc)-1)] + [TTg_direct_trunc[-1].shape[2]]

    if ifEval:
        g_direct = tl.tt_to_tensor(TTg_direct_trunc)
        err_g_direct_trunc = np.linalg.norm(real_g - g_direct) / np.linalg.norm(real_g)
        error_dict_direct[f"rounding rmax={rm}"] = err_g_direct_trunc


print("\n === Direct Method Errors ===\n")
for key, value in error_dict_direct.items():
    print(f"{key}: {value}")
print("\n =========================\n")