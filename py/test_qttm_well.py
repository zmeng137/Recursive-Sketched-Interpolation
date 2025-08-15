import numpy as np
import time as tm
import tensorly as tl
import matplotlib.pyplot as plt

from utils import load_quantics_tensor_hdf5, convert_quantics_tensor_to_1d
from qttm import QTTM_INTCONT_NOEVAL

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MLA-Toolkit', 'py'))
from tensor_cross import TT_IDPRRLDU, TT_CUR_L2R


# Load function 1 and function 2 from hdf5 files 
filepath_f1 = "/home/zmeng5/QTTM/datasets/qtensor_well/qt_MHD64_1.hdf5"
filepath_f2 = "/home/zmeng5/QTTM/datasets/qtensor_well/qt_MHD64_1.hdf5"
qtensor_f1, metadata_f1 = load_quantics_tensor_hdf5(filepath_f1)
qtensor_f2, metadata_f2 = load_quantics_tensor_hdf5(filepath_f2)

r_max_f1 = 200
r_max_f2 = 200
eps = 1e-14

TTCore_f1, _, TTRank_f1, InterpSet_I_f1, _ = TT_CUR_L2R(qtensor_f1, r_max_f1, eps, 0, 0)
TTCore_f2, _, TTRank_f2, InterpSet_I_f2, _ = TT_CUR_L2R(qtensor_f2, r_max_f2, eps, 0, 0)

InterpSet_I_f1[0] = []
InterpSet_I_f2[0] = []

start_t = tm.time()
recon_f1 = tl.tt_to_tensor(TTCore_f1)
recon_f2 = tl.tt_to_tensor(TTCore_f2)
recon_g = recon_f1 * recon_f2  # g derived from RED
TT_CUR_L2R(recon_g, 300, eps, 0, 0)
end_t = tm.time()
elapsed_ms = (end_t - start_t) * 1000
print(f"Elapsed time: {elapsed_ms:.2f} ms")

error_f1 = tl.norm(qtensor_f1 - recon_f1) / tl.norm(qtensor_f1)
error_f2 = tl.norm(qtensor_f2 - recon_f2) / tl.norm(qtensor_f2)
print(f"Relative error: f1 (r_max = {r_max_f1}): {error_f1}; f2 (r_max = {r_max_f2}): {error_f2}")

# Real g = f1 * f2
real_g = qtensor_f1 * qtensor_f2

# Get new g's I, J sets from f1 TCI and f2 TCI via the integral method
contract_number = 9
r_max = 300
start_t = tm.time()
interp_I_g, interp_J_g, TTRank_g, TT_cores_g = QTTM_INTCONT_NOEVAL(
                TTCore_f1, InterpSet_I_f1, TTRank_f1,
                TTCore_f2, InterpSet_I_f2, TTRank_f2,
                contract_number, r_max, eps)
end_t = tm.time()
elapsed_ms = (end_t - start_t) * 1000
print(f"Elapsed time: {elapsed_ms:.2f} ms")

error_vs_real = tl.norm(real_g - tl.tt_to_tensor(TT_cores_g)) / tl.norm(real_g)
error_vs_recon = tl.norm(recon_g - tl.tt_to_tensor(TT_cores_g)) / tl.norm(recon_g)
print(f"Relative error (vs real g) at r_max = {r_max}: {error_vs_real}")
print(f"Relative error (vs recon g) at r_max = {r_max}: {error_vs_recon}")

'''
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
plt.savefig("active_matter_test.png")
'''