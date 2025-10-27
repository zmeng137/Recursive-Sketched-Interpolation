import os
import sys
import numpy as np
import time as tm
import tensorly as tl
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MLA-Toolkit', 'py'))

from tci import TT_IDPRRLDU_L2R
from multiply import multiply_tt
from utils import load_quantics_tensor_hdf5, convert_quantics_tensor_to_1d, size_tt, load_quantics_tensor_formula, tt_to_tensor_tensordot

''' ===================== Tensor Generation ===================== '''

def quantics_function_tensor():
    # Load quantics function tensors from synthetic formulas 
    #qtensor_f1, _ = load_quantics_tensor_formula(0, 15)
    #qtensor_f2, _ = load_quantics_tensor_formula(1, 15)

    # Load quantics function tensors from hdf5 files 
    filePath_f1 = "/home/zmeng5/QTTM/datasets/qtensor_gaussian/mix4d_gaussian_0.hdf5"
    filePath_f2 = "/home/zmeng5/QTTM/datasets/qtensor_gaussian/mix4d_gaussian_1.hdf5"
    qtensor_f1, _ = load_quantics_tensor_hdf5(filePath_f1)
    qtensor_f2, _ = load_quantics_tensor_hdf5(filePath_f2)

    # TT Decomposition of f1 and f2
    r_max_f1 = 25
    r_max_f2 = 25
    eps = 1e-10
    TTCore_f1, _, _ = TT_IDPRRLDU_L2R(qtensor_f1, r_max_f1, eps, 0)
    TTCore_f2, _, _ = TT_IDPRRLDU_L2R(qtensor_f2, r_max_f2, eps, 0)
    
    recon_f1 = tl.tt_to_tensor(TTCore_f1)
    recon_f2 = tl.tt_to_tensor(TTCore_f2)
    error_f1 = tl.norm(qtensor_f1 - recon_f1) / tl.norm(qtensor_f1)
    error_f2 = tl.norm(qtensor_f2 - recon_f2) / tl.norm(qtensor_f2)
    print(f"Relative error of TT_f1 (r_max = {r_max_f1}): {error_f1}")
    print(f"Relative error of TT_f2 (r_max = {r_max_f2}): {error_f2}")
    
    return TTCore_f1, TTCore_f2

def general_synthetic_tensor():
    # Settings
    shape = [7, 3, 4, 5, 4, 10, 5, 2, 3, 5]
    ttrank_tt1 = [1, 4, 5, 11, 8, 30, 11, 25, 8, 3, 1]
    ttrank_tt2 = [1, 3, 9, 10, 17, 23, 19, 31, 9, 2, 1]
    seed_tt1 = 20
    seed_tt2 = 10

    # Random TT
    random_tt1 = tl.random.random_tt(shape, ttrank_tt1, False, seed_tt1)
    random_tt2 = tl.random.random_tt(shape, ttrank_tt2, False, seed_tt2)

    return random_tt1, random_tt2

tt_f1, tt_f2 = quantics_function_tensor()
#tt_f1, tt_f2 = general_synthetic_tensor()

''' ===================== Functional TT Test ===================== '''

# RED
recon_tensor1 = tl.tt_to_tensor(tt_f1)
recon_tensor2 = tl.tt_to_tensor(tt_f2)
g_tensor = recon_tensor1 * recon_tensor2

r_max = 50
eps = 1e-10
TTCore, TTRank, InterpSet_I = TT_IDPRRLDU_L2R(g_tensor, r_max, eps, 0)
error =  tl.norm(g_tensor - tl.tt_to_tensor(TTCore)) / tl.norm(g_tensor)
print(f"Relative error (vs recon_t1 * recon_t2) at r_max = {r_max}: {error}")

# Recursive Sketching Interpolative Algorithm
contract_number = 2
r_max = 50
seed = 10
eps=1e-10
over_sampling = 40
TT_cores_g, TTRank_g, interp_I_gBasis  = multiply_tt(tt_f1, tt_f2, contract_number, r_max, eps, over_sampling, seed)

recon_g_sk = tl.tt_to_tensor(TT_cores_g)
error_vs_real = tl.norm(g_tensor - recon_g_sk) / tl.norm(g_tensor)
print(f"TT-rank of new QTT of g: {TTRank_g}")
print(f"Relative error (vs real g) at r_max = {r_max}: {error_vs_real}")
