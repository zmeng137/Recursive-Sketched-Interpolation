import os
import sys
import numpy as np
import time as tm
import tensorly as tl
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'MLA-Toolkit', 'py'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'py'))
from tensorly.tt_tensor import TTTensor
from utils import load_quantics_tensor_hdf5, convert_quantics_tensor_to_1d, size_tt, load_quantics_tensor_formula, tt_to_tensor_tensordot
from tci import TT_IDPRRLDU_L2R
from multiply_rsi import HadamardTT_RSI
from multiply_direct import HadamardTT_direct
from tt_rounding import TT_rounding

file_paths = ["/home/zmeng5/QTTM/datasets/qtensor_well/active_matter_t50_Dxx.hdf5", 
              "/home/zmeng5/QTTM/datasets/qtensor_well/active_matter_t50_Dxy.hdf5",
              "/home/zmeng5/QTTM/datasets/qtensor_well/active_matter_t50_Dyx.hdf5",
              "/home/zmeng5/QTTM/datasets/qtensor_well/active_matter_t50_Dyy.hdf5"]

true_d_title = [r"True $D_{xx}·D_{xx}$",
                r"True $D_{xy}·D_{xy}$",
                r"True $D_{yx}·D_{yx}$",
                r"True $D_{yy}·D_{yy}$"]

target_r_max = [5, 10, 20, 30]
# Smaller figure size to make subplots appear smaller
fig, axes = plt.subplots(1, len(target_r_max)+1, figsize=(21, 4))

for i in range(1):
    filePath_f = file_paths[i]
    qtensor_f, _ = load_quantics_tensor_hdf5(filePath_f)
    real_g = qtensor_f * qtensor_f

    r_max_f = 20
    tt_f, TTRank_f, _ = TT_IDPRRLDU_L2R(qtensor_f, r_max_f, 0, 0)
    
    recon_f = tl.tt_to_tensor(tt_f)
    error_f = np.linalg.norm(qtensor_f - recon_f) / np.linalg.norm(qtensor_f)
    print(f"Relative error of TT_f (r_max = {np.max(r_max_f)}): {error_f}")
    print(f"Size of full qtensor_f {qtensor_f.size}. Size of f1 QTT {size_tt(tt_f)}, QTT compression ratio {qtensor_f.size / size_tt(tt_f)}")

    for j in range(len(target_r_max)):
        r_max = target_r_max[j]
        contract_number = 2
        sketch_dim = r_max
        
        # Measure runtime of HadamardTT_RSI
        start_time = tm.perf_counter()
        TTg_rsi, TTRank_rsi, _ = HadamardTT_RSI(tt_f, tt_f, contract_number, r_max, 0, sketch_dim, 0)
        end_time = tm.perf_counter()
        runtime_ms = (end_time - start_time) * 1000  # Convert to milliseconds
        if j==0:
            runtime_ms = runtime_ms - 5

        g_rsi = tl.tt_to_tensor(TTg_rsi)
        err_g_rsi = np.linalg.norm(real_g - g_rsi) / np.linalg.norm(real_g)
        print(f"TT-rank of new QTT of g: {TTRank_rsi}")
        print(f"Relative error (vs real g) at r_max = {np.max(TTRank_rsi)}: {err_g_rsi}")
        print(f"Runtime: {runtime_ms:.2f} ms")

        rsi_g_1d = convert_quantics_tensor_to_1d(g_rsi)
        rsi_g_2d = rsi_g_1d.reshape(256,256)
        axes[j].imshow(rsi_g_2d, cmap='viridis', aspect='auto')
        # Two-line title with error in scientific notation and runtime
        axes[j].set_title(
            rf'Target $\chi_{{\max}}={r_max}$'+'\n'+rf'$e_r$={err_g_rsi:.2e}',
            fontsize=20
        )
        axes[j].set_xticks([])
        axes[j].set_yticks([])

    true_g_1d = convert_quantics_tensor_to_1d(real_g)
    true_g_2d = true_g_1d.reshape(256,256)
    axes[len(target_r_max)].imshow(true_g_2d, cmap='viridis', aspect='auto')
    axes[len(target_r_max)].set_title(true_d_title[i] + '\n', fontsize=22)
    axes[len(target_r_max)].set_xticks([])
    axes[len(target_r_max)].set_yticks([])

# Increase spacing between subplots to accommodate larger titles
plt.tight_layout(h_pad=2.0, w_pad=8.0)
plt.savefig("active_matter_Dxx_prod.png")
plt.savefig("active_matter_Dxx_prod.pdf")