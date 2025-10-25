import os
import sys
import numpy as np
import time as tm
import tensorly as tl
import matplotlib.pyplot as plt

#from utils import load_quantics_tensor_hdf5, convert_quantics_tensor_to_1d, size_tt, load_quantics_tensor_formula, tt_to_tensor_tensordot
from multiply import multiply_tt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MLA-Toolkit', 'py'))
from tci import TT_IDPRRLDU_L2R

shape = [7, 3, 4, 5, 4, 10, 5, 2, 3, 5]

ttrank_tt1 = [1, 4, 5, 11, 8, 10, 11, 25, 8, 3, 1]
ttrank_tt2 = [1, 3, 9, 10, 13, 21, 19, 20, 9, 2, 1]

seed_tt1 = 20
seed_tt2 = 10
random_tt1 = tl.random.random_tt(shape, ttrank_tt1, False, seed_tt1)
random_tt2 = tl.random.random_tt(shape, ttrank_tt2, False, seed_tt2)

# RED
syn_tensor1 = tl.tt_to_tensor(random_tt1)
syn_tensor2 = tl.tt_to_tensor(random_tt2)
g_tensor = syn_tensor1 * syn_tensor2

r_max = 40
eps = 1e-10
TTCore, TTRank, InterpSet_I = TT_IDPRRLDU_L2R(g_tensor, r_max, eps, 0)
error =  tl.norm(g_tensor - tl.tt_to_tensor(TTCore)) / tl.norm(g_tensor)
print(f"Relative error (vs real f) at r_max = {r_max}: {error}")

# RSI
contract_number = 3
r_max = 40
randomFlag = 1
seed = 0
eps=1e-10
skLayer = 5
interp_I_g, TTRank_g, TT_cores_g = multiply_tt(random_tt1, random_tt2, contract_number, r_max, eps, randomFlag, seed, skLayer)

recon_g_sk = tl.tt_to_tensor(TT_cores_g)
error_vs_real = tl.norm(g_tensor - recon_g_sk) / tl.norm(g_tensor)
print(f"TT-rank of new QTT of g: {TTRank_g}")
print(f"Relative error (vs real g) at r_max = {r_max}: {error_vs_real}")
