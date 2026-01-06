import os
import sys
import numpy as np
import time as tm
import tensorly as tl
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MLA-Toolkit', 'py'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'py'))

from tci import TT_IDPRRLDU_L2R
from map_rsi import NonlinearMapTT_RSI
from utils import load_quantics_tensor_hdf5, convert_quantics_tensor_to_1d, size_tt, load_quantics_tensor_formula, tt_to_tensor_tensordot

''' ===================== Tensor Generation ===================== '''

def quantics_function_tensor():
    # Load quantics function tensors from synthetic formulas 
    #qtensor_f1, _ = load_quantics_tensor_formula(0, 15)
    #qtensor_f2, _ = load_quantics_tensor_formula(1, 15)

    # Load quantics function tensors from hdf5 files 
    #filePath_f1 = "/home/zmeng5/QTTM/datasets/qtensor_gaussian/mix4d_gaussian_0.hdf5"
    filePath_f1 = "/home/zmeng5/QTTM/datasets/qtensor_well/qt_active_matter_9.hdf5"
    qtensor_f, _ = load_quantics_tensor_hdf5(filePath_f1)

    # TT Decomposition of f
    r_max_f = 10
    eps = 1e-10
    TTCore_f, _, _ = TT_IDPRRLDU_L2R(qtensor_f, r_max_f, eps, 0)
    recon_f = tl.tt_to_tensor(TTCore_f)
    error_f = tl.norm(qtensor_f - recon_f) / tl.norm(qtensor_f)
    print(f"Relative error of TT_f (r_max = {r_max_f}): {error_f}")
    print(f"Size of full qtensor_f {qtensor_f.size}. Size of f1 QTT {size_tt(TTCore_f)}, QTT compression ratio {qtensor_f.size / size_tt(TTCore_f)}")
    return TTCore_f

def general_synthetic_tensor():
    # Settings
    shape = [5, 5, 5, 5, 5, 5, 5, 5]
    ttrank = [1, 4, 11, 8, 10, 25, 8, 3, 1]
    seed = 10

    # Random TT
    random_tt = tl.random.random_tt(shape, ttrank, False, seed)
     
    return random_tt

print("Generating test tensor...")
tt_f = general_synthetic_tensor()
print("Tensor generated.")

''' ===================== Functional TT Test ===================== '''

# Function g to be applied to the tensor network
#g_func = lambda x: np.cos(-2*x) * x + 2 - np.sin(3 * x) - x**3 / 100 + np.exp(-x*x/10)
#g_func = lambda x: x ** 2
#g_func = lambda x: np.maximum(0, x)
g_func = lambda x: 1 / (1 + np.exp(-x))  # sigmoid function
g_func = lambda x: np.maximum(0,x)           # ReLU function

# RED
r_max = 30
eps = 1e-10
recon_f = tl.tt_to_tensor(tt_f)
tensor_g = g_func(recon_f)
TTCore_g, _, _ = TT_IDPRRLDU_L2R(tensor_g, r_max, eps, 0)
error_g =  tl.norm(tensor_g - tl.tt_to_tensor(TTCore_g)) / tl.norm(tensor_g)
print(f"Relative error (vs g[recon_f]) at r_max = {r_max}: {error_g}")

# Recursive Sketching Interpolative Algorithm
contract_number = 2
r_max = 30
seed = 10
eps=1e-10
over_sampling = r_max
TT_cores_g, TTRank_g, interp_I_gBasis = NonlinearMapTT_RSI(g_func, tt_f, contract_number, r_max, eps, over_sampling, seed)

recon_g_sk = tl.tt_to_tensor(TT_cores_g)
error_vs_real = tl.norm(tensor_g - recon_g_sk) / tl.norm(tensor_g)
print(f"TT-rank of new QTT of g: {TTRank_g}")
print(f"Relative error (vs real g) at r_max = {r_max}: {error_vs_real}")
