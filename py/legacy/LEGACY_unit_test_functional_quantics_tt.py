import os
import sys
import numpy as np
import time as tm
import tensorly as tl
import matplotlib.pyplot as plt

from utils import load_quantics_tensor_hdf5, convert_quantics_tensor_to_1d, size_tt, load_quantics_tensor_formula, tt_to_tensor_tensordot
from functional import functional_tt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MLA-Toolkit', 'py'))
from tci import TT_IDPRRLDU_L2R

''' Tensor Construction '''
# Load quantics function tensors from synthetic formulas 
#qtensor_f1, _ = load_quantics_tensor_formula(0, 15)
#qtensor_f2, _ = load_quantics_tensor_formula(1, 15)

# Load quantics function tensors from hdf5 files 
filePath_f1 = "/home/zmeng5/QTTM/datasets/qtensor_gaussian/mix4d_gaussian_0.hdf5"
qtensor_f, metadata_f = load_quantics_tensor_hdf5(filePath_f1)

g_func = lambda x: x ** 2 - 3 * x + 10 + 0.4 * np.exp(x)
#g_func = lambda x: np.cos(-np.exp(-x*x/12*np.pi) + x**3 - 12 + 15.2*np.sin(10*np.pi*x)*np.exp(x-1))

# Real g = f1 * f2
real_g = g_func(qtensor_f)


''' Tensor Train Decomposition '''
# TT-CUR Left->Right (One side)
r_max_f = 40
eps = 1e-10
TTCore_f, TTRank_f, InterpSet_I_f = TT_IDPRRLDU_L2R(qtensor_f, r_max_f, eps, 0)
InterpSet_I_f[0] = []

# Reconstruction test
recon_f = tl.tt_to_tensor(TTCore_f)
error_f = tl.norm(qtensor_f - recon_f) / tl.norm(qtensor_f)
print(f"Relative error: f1 (r_max = {r_max_f}): {error_f}")

# QTTM: RED method
# (i) Reconstruction
start_t_recon = tm.time()
r_max = 40
recon_f = tt_to_tensor_tensordot(TTCore_f)
end_t_recon = tm.time()

# (ii) Evaluation (Multiplication)
start_t_eval = tm.time()
recon_g = g_func(recon_f)  # g derived from RED
end_t_eval = tm.time()

# (iii) Decomposition 
start_t_decomp = tm.time()
TTCore_g_red, TTRank_g_red, InterpSet_I_g_red= TT_IDPRRLDU_L2R(recon_g, r_max, eps, 0)
end_t_decomp = tm.time()

elapsed_ms_recon = (end_t_recon - start_t_recon) * 1000
elapsed_ms_eval = (end_t_eval - start_t_eval) * 1000
elapsed_ms_decomp = (end_t_decomp - start_t_decomp) * 1000
print(f"Elapsed time of RED: Recon {elapsed_ms_recon:.2f} ms + Eval {elapsed_ms_eval:.2f} ms + Decomp {elapsed_ms_decomp:.2f} ms = {(elapsed_ms_recon + elapsed_ms_eval + elapsed_ms_decomp):.2f} ms")

# Check accuracy
recon_g_red = tl.tt_to_tensor(TTCore_g_red)
error_g_red_recon = tl.norm(recon_g - recon_g_red) / tl.norm(recon_g)
error_g_red_real =  tl.norm(real_g - recon_g_red) / tl.norm(real_g)
print(f"TT-rank of new QTT of g: {TTRank_g_red}")
print(f"Relative error (vs real g) at r_max = {r_max}: {error_g_red_real}")
print(f"Relative error (vs recon g) at r_max = {r_max}: {error_g_red_recon}")


# Get new g's I, J sets from f1 TCI and f2 TCI via the integral method
contract_number = 2
r_max = 40
seed = 10
over_sampling = 40
interp_I_g, TTRank_g, TT_cores_g = functional_tt(g_func, TTCore_f, contract_number, r_max, eps, over_sampling, seed)

recon_g_sk = tl.tt_to_tensor(TT_cores_g)
error_vs_real = tl.norm(real_g - recon_g_sk) / tl.norm(real_g)
error_vs_recon = tl.norm(recon_g - recon_g_sk) / tl.norm(recon_g)
print(f"TT-rank of new QTT of g: {TTRank_g}")
print(f"Relative error (vs real g) at r_max = {r_max}: {error_vs_real}")
print(f"Relative error (vs recon g) at r_max = {r_max}: {error_vs_recon}")

'''
# Plot the 1D flattened Gaussian function
realg_1d = convert_quantics_tensor_to_1d(real_g)
qttmg_1d = convert_quantics_tensor_to_1d(recon_g_sk)
plt.figure(figsize=(12, 6))
plt.plot(realg_1d, linewidth=0.5, label='real g(f(x))')
plt.plot(qttmg_1d, linewidth=0.5, label='g(f(x)) from our method')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'1D Flattened 4D Gaussian Function (Total points: {len(real_g)})')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('1d_flattened_gaussian.png', dpi=150)
'''