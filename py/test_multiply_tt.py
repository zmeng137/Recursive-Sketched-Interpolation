import os
import sys
import numpy as np
import time as tm
import tensorly as tl
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MLA-Toolkit', 'py'))
from tensorly.tt_tensor import TTTensor
from utils import load_quantics_tensor_hdf5, convert_quantics_tensor_to_1d, size_tt, load_quantics_tensor_formula, tt_to_tensor_tensordot
from tci import TT_IDPRRLDU_L2R       # One-side/sweep TCI (TT-ID)
from multiply_rsi import HadamardTT_RSI   # RSI method for TT product
from multiply_direct import HadamardTT_direct  # Direct method for TT product
from tt_rounding import TT_rounding   # TT rounding for rank compression

''' ===================== Tensor Generation ===================== '''

def quantics_function_tensor():
    # Load quantics function tensors from synthetic formulas 
    #qtensor_f1, _ = load_quantics_tensor_formula(7, 20)
    #qtensor_f2, _ = load_quantics_tensor_formula(8, 20)


    # Load quantics function tensors from hdf5 files 
    filePath_f1 = "/home/zmeng5/QTTM/datasets/qtensor_well/active_matter_t50_c.hdf5"
    filePath_f2 = "/home/zmeng5/QTTM/datasets/qtensor_well/active_matter_t50_c.hdf5"
    qtensor_f1, _ = load_quantics_tensor_hdf5(filePath_f1)
    qtensor_f2, _ = load_quantics_tensor_hdf5(filePath_f2)

    real_g = qtensor_f1 * qtensor_f2

    # TT Decomposition of f1 and f2
    r_max_f1 = 100
    r_max_f2 = 100
    eps = 0

    #TTCore_f1, TTRank_f1, _ = TT_IDPRRLDU_L2R(qtensor_f1, r_max_f1, eps, 0)
    #TTCore_f2, TTRank_f2, _ = TT_IDPRRLDU_L2R(qtensor_f2, r_max_f2, eps, 0)
    TTCore_f1 = tl.decomposition.tensor_train(qtensor_f1, r_max_f1)
    TTCore_f2 = tl.decomposition.tensor_train(qtensor_f2, r_max_f2)
    
    
    recon_f1 = tl.tt_to_tensor(TTCore_f1)
    recon_f2 = tl.tt_to_tensor(TTCore_f2)
    error_f1 = tl.norm(qtensor_f1 - recon_f1) / tl.norm(qtensor_f1)
    error_f2 = tl.norm(qtensor_f2 - recon_f2) / tl.norm(qtensor_f2)
    print(f"Relative error of TT_f1 (r_max = {np.max(r_max_f1)}): {error_f1}")
    print(f"Relative error of TT_f2 (r_max = {np.max(r_max_f1)}): {error_f2}")
    print(f"Size of full qtensor_f {qtensor_f1.size}. Size of f1 QTT {size_tt(TTCore_f1)}, QTT compression ratio {qtensor_f1.size / size_tt(TTCore_f1)}")
    print(f"Size of full qtensor_f {qtensor_f2.size}. Size of f1 QTT {size_tt(TTCore_f2)}, QTT compression ratio {qtensor_f2.size / size_tt(TTCore_f2)}")
    

    ifPlot = False
    if ifPlot:
        f1_1d = convert_quantics_tensor_to_1d(qtensor_f1)
        f2_1d = convert_quantics_tensor_to_1d(qtensor_f2)
        ngrid = len(f1_1d)
        xgrid = np.linspace(0, 1, ngrid)
        plt.figure(figsize=(10,4))
        plt.plot(xgrid, f1_1d, label='f1(x)', linewidth=2)
        plt.plot(xgrid, f2_1d, label='f2(x)', linewidth=2)
        plt.xlabel('x', fontsize=14)
        plt.ylabel('Function value', fontsize=14)
        plt.title('1D Functions from Quantics Tensors', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid()
        plt.savefig('gaussian_1d_plot.png', dpi=300)

    return TTCore_f1, TTCore_f2, real_g

def quantics_random_tt():
    digit = 50
    shape = [2 for i in range(digit)]

    #ttrank_tt1 = [1, 2, 4, 8] + [10 for i in range(digit - 6 - 1)] + [8, 4, 2, 1]
    #ttrank_tt2 = [1, 2, 4, 8] + [10 for i in range(digit - 6 - 1)] + [8, 4, 2, 1]
    r_max = 60
    ttrank_tt1 = [1] + [r_max for i in range(digit - 1)] + [1]
    ttrank_tt2 = [1] + [r_max for i in range(digit - 1)] + [1]

    seed_tt1 = 12
    seed_tt2 = 98

    # Random TT
    random_tt1 = tl.random.random_tt(shape, ttrank_tt1, False, seed_tt1)
    random_tt2 = tl.random.random_tt(shape, ttrank_tt2, False, seed_tt2)
    return random_tt1, random_tt2

def general_synthetic_tensor():
    # Settings
    shape = [20, 20, 20, 20, 20]
    ttrank_tt1 = [1, 20, 100, 100, 20, 1]
    ttrank_tt2 = [1, 20, 100, 100, 20, 1]

    seed_tt1 = 12
    seed_tt2 = 98

    # Random TT
    random_tt1 = tl.random.random_tt(shape, ttrank_tt1, False, seed_tt1)
    random_tt2 = tl.random.random_tt(shape, ttrank_tt2, False, seed_tt2)
    real_g = tl.tt_to_tensor(random_tt1) * tl.tt_to_tensor(random_tt2)

    return random_tt1, random_tt2, real_g

tt_f1, tt_f2, real_g = quantics_function_tensor()
#tt_f1, tt_f2, real_g = general_synthetic_tensor()
#tt_f1, tt_f2 = quantics_random_tt()

''' ===================== TT Multiplication ===================== '''
print("\n === Hadamard Product Test ===\n")

# RED
#recon_tensor1 = tl.tt_to_tensor(tt_f1)
#recon_tensor2 = tl.tt_to_tensor(tt_f2)
#g_tensor = recon_tensor1 * recon_tensor2

#r_max = 10
#eps = 1e-15
#TTCore, TTRank, InterpSet_I = TT_IDPRRLDU_L2R(g_tensor, r_max, eps, 0)
#error =  tl.norm(real_g - tl.tt_to_tensor(TTCore)) / tl.norm(real_g)
#print(f"Relative error (vs recon_t1 * recon_t2) at r_max = {np.max(TTRank)}: {error}")

# Recursive Sketching Interpolation (RSI) Algorithm
contract_number = 2
r_max = 50
seed = 10
eps=0
over_sampling = 50
TTg_rsi, TTRank_rsi, _  = HadamardTT_RSI(tt_f1, tt_f2, contract_number, r_max, eps, over_sampling, seed)

# Direct method
TTg_direct = HadamardTT_direct(tt_f1, tt_f2)
TTrank_direct = [TTg_direct[k].shape[2] for k in range(len(TTg_direct)-1)]
TTg_direct_trunc = TT_rounding(TTg_direct, 1e-16, r_max)   # Post re-compression
TTrank_direct_trunc = [TTg_direct_trunc[k].shape[2] for k in range(len(TTg_direct_trunc)-1)]


''' ===================== Result Evaluation ===================== '''
print("\n === Result Evaluation ===\n")

ifEval = True
if ifEval:
    # Evaluate RSI quality
    g_rsi = tl.tt_to_tensor(TTg_rsi)
    err_g_rsi = tl.norm(real_g - g_rsi, 2) / tl.norm(real_g, 2)
    print(f"TT-rank of new QTT of g: {TTRank_rsi}")
    print(f"Relative error (vs real g) at r_max = {np.max(TTRank_rsi)}: {err_g_rsi}")

    # Evaluate Direct quality
    g_direct = tl.tt_to_tensor(TTg_direct)
    err_g_direct = tl.norm(real_g - g_direct, 2) / tl.norm(real_g, 2)
    print(f"TT-rank of new QTT of g: {TTrank_direct}")
    print(f"Relative error (vs real g) at r_max = {np.max(TTrank_direct)}: {err_g_direct}")

    # Evaluate Compressed Direct quality
    g_direct_trunc = tl.tt_to_tensor(TTg_direct_trunc)
    err_g_direct_trunc = tl.norm(real_g - g_direct_trunc, 2) / tl.norm(real_g, 2)
    print(f"TT-rank of new QTT of g: {TTrank_direct_trunc}")
    print(f"Relative error (vs real g) at r_max = {np.max(TTrank_direct_trunc)}: {err_g_direct_trunc}")
    

ifPlot = False 
if ifPlot:
    g_1d = convert_quantics_tensor_to_1d(real_g)
    g_sk_1d = convert_quantics_tensor_to_1d(g_rsi)
    ngrid = len(g_1d)
    xgrid = np.linspace(0, 1, ngrid)
    plt.figure(figsize=(10,4))
    plt.plot(xgrid, g_1d, label='g(x) real', linewidth=2)
    plt.plot(xgrid, g_sk_1d, label='g(x) sketched', linestyle='--', linewidth=2)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('Function value', fontsize=14)
    plt.title('1D Function g from Quantics Tensor', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid()
    plt.savefig('g_1d_plot.png', dpi=300)
