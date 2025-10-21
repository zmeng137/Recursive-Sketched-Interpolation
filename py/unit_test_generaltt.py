import os
import sys
import numpy as np
import time as tm
import tensorly as tl
import matplotlib.pyplot as plt

from utils import load_quantics_tensor_hdf5, convert_quantics_tensor_to_1d, size_tt, load_quantics_tensor_formula, tt_to_tensor_tensordot
from functional import functional_qtt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MLA-Toolkit', 'py'))
from tci import TT_IDPRRLDU_L2R

g_func = lambda x: -2*x + 2 - np.sin(3 * x) - x**3 / 100 + np.exp(-x*x/10)

shape = [10, 20, 10, 30, 10]
ttrank = [1, 7, 30, 40, 5, 1]
seed = 1
random_tt = tl.random.random_tt(shape, ttrank, False, seed)
syn_tensor = tl.tt_to_tensor(random_tt)
syn_tensor_g = g_func(syn_tensor)

r_max = 40
eps = 1e-10
TTCore, TTRank, InterpSet_I = TT_IDPRRLDU_L2R(syn_tensor, r_max, eps, 0)
error =  tl.norm(syn_tensor - tl.tt_to_tensor(TTCore)) / tl.norm(syn_tensor)
print(f"Relative error (vs real f) at r_max = {r_max}: {error}")

r_max = 40
eps = 1e-10
TTCore_g, TTRank_g, InterpSet_I_g = TT_IDPRRLDU_L2R(syn_tensor_g, r_max, eps, 0)
error_g =  tl.norm(syn_tensor_g - tl.tt_to_tensor(TTCore_g)) / tl.norm(syn_tensor_g)
print(f"Relative error (vs real g[f]) at r_max = {r_max}: {error_g}")

InterpSet_I[0] = []
contract_number = 2
r_max = 50
randomFlag = 1
seed = 0
skLayer = 30

interp_I_g, TTRank_g, TT_cores_g = functional_qtt(g_func,
                TTCore, InterpSet_I, TTRank,
                contract_number, r_max, eps,
                randomFlag, seed, skLayer)

recon_g_sk = tl.tt_to_tensor(TT_cores_g)
error_vs_real = tl.norm(syn_tensor_g - recon_g_sk) / tl.norm(syn_tensor_g)
print(f"TT-rank of new QTT of g: {TTRank_g}")
print(f"Relative error (vs real g) at r_max = {r_max}: {error_vs_real}")
