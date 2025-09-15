import os
import sys
import numpy as np
import time as tm
import tensorly as tl
import matplotlib.pyplot as plt

from utils import load_quantics_tensor_hdf5, convert_quantics_tensor_to_1d, size_tt, load_quantics_tensor_formula, tt_to_tensor_tensordot
from qttm import qttm_intcont_noeval, qttm_ric, tt_contraction_opcount

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MLA-Toolkit', 'py'))
from tci import TT_CUR_L2R

''' Tensor Construction '''
# Load quantics function tensors from synthetic formulas 
#qtensor_f1, _ = load_quantics_tensor_formula(0, 15)
#qtensor_f2, _ = load_quantics_tensor_formula(1, 15)

# Load quantics function tensors from hdf5 files 
filePath_f1 = "/home/zmeng5/QTTM/datasets/qtensor_gaussian/mix2d_gaussian_0.hdf5"
filePath_f2 = "/home/zmeng5/QTTM/datasets/qtensor_gaussian/mix2d_gaussian_1.hdf5"
qtensor_f1, metadata_f1 = load_quantics_tensor_hdf5(filePath_f1)
qtensor_f2, metadata_f2 = load_quantics_tensor_hdf5(filePath_f2)

# Real g = f1 * f2
real_g = qtensor_f1 * qtensor_f2


''' Tensor Train Decomposition '''
# TT-CUR Left->Right (One side)
r_max_f1 = 15
r_max_f2 = 15
eps = 1e-14
TTCore_f1, _, TTRank_f1, InterpSet_I_f1, _ = TT_CUR_L2R(qtensor_f1, r_max_f1, eps, 0, 0)
TTCore_f2, _, TTRank_f2, InterpSet_I_f2, _ = TT_CUR_L2R(qtensor_f2, r_max_f2, eps, 0, 0)
InterpSet_I_f1[0] = []
InterpSet_I_f2[0] = []

# Statistics for TT reconstruction
recon_op_f1, recon_sz_f1 = tt_contraction_opcount(TTCore_f1)
recon_op_f2, recon_sz_f2 = tt_contraction_opcount(TTCore_f2)
print(f"Size of f1 full tensor {qtensor_f1.size}. Size of f1 QTT {size_tt(TTCore_f1)}")
print(f"Size of f2 full tensor {qtensor_f2.size}. Size of f2 QTT {size_tt(TTCore_f2)}")
print(f"Reconstruction statistics of TTCore_f1 -- Total ops: {recon_op_f1}, Max size: {recon_sz_f1}")
print(f"Reconstruction statistics of TTCore_f2 -- Total ops: {recon_op_f2}, Max size: {recon_sz_f2}")

# Reconstruction test
recon_f1 = tl.tt_to_tensor(TTCore_f1)
recon_f2 = tl.tt_to_tensor(TTCore_f2)
error_f1 = tl.norm(qtensor_f1 - recon_f1) / tl.norm(qtensor_f1)
error_f2 = tl.norm(qtensor_f2 - recon_f2) / tl.norm(qtensor_f2)
print(f"Relative error: f1 (r_max = {r_max_f1}): {error_f1}; f2 (r_max = {r_max_f2}): {error_f2}")

# QTTM: RED method
# (i) Reconstruction
start_t_recon = tm.time()
r_max = 30
recon_f1 = tt_to_tensor_tensordot(TTCore_f1)
recon_f2 = tt_to_tensor_tensordot(TTCore_f2)
end_t_recon = tm.time()

# (ii) Evaluation (Multiplication)
start_t_eval = tm.time()
recon_g = recon_f1 * recon_f2  # g derived from RED
end_t_eval = tm.time()

# (iii) Decomposition 
start_t_decomp = tm.time()
TTCore_g_red, _, TTRank_g_red, InterpSet_I_g_red, _ = TT_CUR_L2R(recon_g, r_max, eps, 0, 0)
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
contract_number = 5
r_max = 20
randomFlag = 1
seed = 0
skLayer = 3

#interp_I_g, interp_J_g, TTRank_g, TT_cores_g = qttm_intcont_noeval(
#                TTCore_f1, InterpSet_I_f1, TTRank_f1,
#                TTCore_f2, InterpSet_I_f2, TTRank_f2,
#                contract_number, r_max, eps)

interp_I_g, TTRank_g, TT_cores_g = qttm_ric(
                TTCore_f1, InterpSet_I_f1, TTRank_f1,
                TTCore_f2, InterpSet_I_f2, TTRank_f2,
                contract_number, r_max, eps,
                randomFlag, seed, skLayer)

error_vs_real = tl.norm(real_g - tl.tt_to_tensor(TT_cores_g)) / tl.norm(real_g)
error_vs_recon = tl.norm(recon_g - tl.tt_to_tensor(TT_cores_g)) / tl.norm(recon_g)
print(f"TT-rank of new QTT of g: {TTRank_g}")
print(f"Relative error (vs real g) at r_max = {r_max}: {error_vs_real}")
print(f"Relative error (vs recon g) at r_max = {r_max}: {error_vs_recon}")


# ====== Plot f1, f2, g ====== 
f1_1d = convert_quantics_tensor_to_1d(qtensor_f1)
f2_1d = convert_quantics_tensor_to_1d(qtensor_f2)
realg_1d = convert_quantics_tensor_to_1d(real_g)
qttmg_1d = convert_quantics_tensor_to_1d(tl.tt_to_tensor(TT_cores_g))

shape2d = [256, 256]
f1_mat2d = f1_1d.reshape(shape2d)
f2_mat2d = f2_1d.reshape(shape2d)
realg_mat2d = realg_1d.reshape(shape2d)
qttmg_mat2d = qttmg_1d.reshape(shape2d)

fig, axes = plt.subplots(2, 2, figsize=(15, 15))

# Original 2D function of f1
im1 = axes[0,0].imshow(f1_mat2d, cmap='viridis')
axes[0,0].set_title('f1 function in 2D')
axes[0,0].set_xlabel('Y index')
axes[0,0].set_ylabel('X index')
plt.colorbar(im1, ax=axes[0,0])

# Original 2D function of f2
im2 = axes[0,1].imshow(f2_mat2d, cmap='viridis')
axes[0,1].set_title('f2 function in 2D')
axes[0,1].set_xlabel('Y index')
axes[0,1].set_ylabel('X index')
plt.colorbar(im2, ax=axes[0,1])

# Original 2D function of f1*f2
im3 = axes[1,0].imshow(realg_mat2d, cmap='viridis')
axes[1,0].set_title('real g=f1*f2 in 2D')
axes[1,0].set_xlabel('Y index')
axes[1,0].set_ylabel('X index')
plt.colorbar(im3, ax=axes[1,0])

# Approximate f1*f2 via QTTM
im4 = axes[1,1].imshow(qttmg_mat2d, cmap='viridis')
axes[1,1].set_title('qttm g=f11*f2 in 2D')
axes[1,1].set_xlabel('Y index')
axes[1,1].set_ylabel('X index')
plt.colorbar(im4, ax=axes[1,1])

plt.tight_layout()
plt.savefig("gaussian_test.png")
