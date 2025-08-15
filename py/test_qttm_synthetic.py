import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt

from qtt import quantics_generation, scatter_plot_f1f2, plot_interp_pivots, QTT_randomSketch
from qttm import QTTM_INTCONT, QTTM_INTCONT_NOEVAL
from utils import Function_Collection

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MLA-Toolkit', 'py'))
from tensor_cross import TT_CUR_L2R, TCI_2site, nested_initIJ_gen_rank1

''' === Quantics representation construction === '''
# Quantics construction
func1 = Function_Collection[3]
func2 = Function_Collection[4]
dim = 12
x_tensor, f1_tensor = quantics_generation(func1, dim)
_,        f2_tensor = quantics_generation(func2, dim)
g_tensor = f1_tensor * f2_tensor
scatter_plot_f1f2(x_tensor, g_tensor, f1_tensor, f2_tensor)

''' === TT-ID for f1, f2, g === '''
r_max = 5
eps = 1e-16
TTCore_f1, _, TTRank_f1, InterpSet_I_f1, _ = TT_CUR_L2R(f1_tensor, r_max, eps, 0, 0)
TTCore_f2, _, TTRank_f2, InterpSet_I_f2, _ = TT_CUR_L2R(f2_tensor, r_max, eps, 0, 0)

recon_f1 = tl.tt_to_tensor(TTCore_f1)
recon_f2 = tl.tt_to_tensor(TTCore_f2)
error_f1 = tl.norm(f1_tensor - recon_f1) / tl.norm(f1_tensor)
error_f2 = tl.norm(f2_tensor - recon_f2) / tl.norm(f2_tensor)
print(f"Relative error of f1 QTT at r_max = {r_max}: {error_f1}, f2 QTT at r_max = {r_max}: {error_f2}")


contract_core_number = 4
r_max = 8
InterpSet_I_f1[0] = []
InterpSet_I_f2[0] = []
interp_I_g, interp_J_g, TTRank_g, TT_cores_g = QTTM_INTCONT_NOEVAL(
                        TTCore_f1, InterpSet_I_f1, TTRank_f1,
                        TTCore_f2, InterpSet_I_f2, TTRank_f2,
                        contract_core_number, r_max, eps)

error_vs_real = tl.norm(recon_f1 * recon_f2 - tl.tt_to_tensor(TT_cores_g)) / tl.norm(recon_f1 * recon_f2)
print(f"Relative error (vs recon g) at r_max = {r_max}: {error_vs_real}")


pass
# Why the error cannot keep going down? Even when I expand the contraction core number...
# Solved -> compare with recon_f1 * recon_f2







''' === Tensor cross interpolation for f1, f2, g === '''
# Create initial (rank-1) interpolation I/J sets
Nested_I_rank1, Nested_J_rank1 = nested_initIJ_gen_rank1(dim)

# TCI-2site of f1
eps = 1e-8
r_max = 7
TT_cross_f1, TT_cores_f1, TTRank_f1, pr_set_f1, pc_set_f1, interp_I_f1, interp_J_f1 = TCI_2site(f1_tensor, eps, r_max, Nested_I_rank1, Nested_J_rank1)
recon_f1 = tl.tt_to_tensor(TT_cores_f1)
error = tl.norm(f1_tensor - recon_f1) / tl.norm(f1_tensor)
print(f"Relative error of f1 QTT at r_max = {r_max}: {error}")

plot_interp_pivots(interp_I_f1, interp_J_f1, x_tensor, f1_tensor)

# TCI-2site of f2
r_max = 5
TT_cross_f2, TT_cores_f2, TTRank_f2, pr_set_f2, pc_set_f2, interp_I_f2, interp_J_f2 = TCI_2site(f2_tensor, eps, r_max, Nested_I_rank1, Nested_J_rank1)
recon_f2 = tl.tt_to_tensor(TT_cores_f2)
error = tl.norm(f2_tensor - recon_f2) / tl.norm(f2_tensor)
print(f"Relative error of f2 QTT at r_max = {r_max}: {error}")

# TCI-2site of g
r_max = 12
TT_cross_g, TT_cores_g, TTRank_g, pr_set_g, pc_set_g, interp_I_g, interp_J_g = TCI_2site(g_tensor, eps, r_max, Nested_I_rank1, Nested_J_rank1)
recon_g = tl.tt_to_tensor(TT_cores_g)
error = tl.norm(g_tensor - recon_g) / tl.norm(g_tensor)
print(f"Relative error of g QTT at r_max = {r_max}: {error}")

''' === QTTM tests === '''
contract_number = 5
r_max = 12
eps = 1e-10

# Get new g's I, J sets from f1 TCI and f2 TCI via the integral method
interp_I_g_new, interp_J_g_new, TTRank_g_new, TT_cores_g_new = QTTM_INTCONT_NOEVAL(
            TT_cores_f1, interp_I_f1, interp_J_f1, TTRank_f1,
            TT_cores_f2, interp_I_f2, interp_J_f2, TTRank_f2,
            contract_number, r_max, eps)

error = tl.norm(g_tensor - tl.tt_to_tensor(TT_cores_g_new)) / tl.norm(g_tensor)
print(f"Relative error of g new at r_max = {r_max}: {error}")

interp_I_g_new, interp_J_g_new, TTRank_g_new, TT_cores_g_new = QTTM_INTCONT(
            f1_tensor, interp_I_f1, interp_J_f1, TTRank_f1, pr_set_f1,
            f2_tensor, interp_I_f2, interp_J_f2, TTRank_f2, pr_set_f2,
            contract_number, r_max, eps)

error = tl.norm(g_tensor - tl.tt_to_tensor(TT_cores_g_new)) / tl.norm(g_tensor)
print(f"Relative error of g new at r_max = {r_max}: {error}")

''' === Limitation of RED - Reconstruction, Evaluation, Decomposition === '''
def red_test():
    print("=== Limitation of RED - Reconstruction, Evaluation, Decomposition ===")

    rerr_f1 = tl.norm(f1_tensor - recon_f1) / tl.norm(f1_tensor)
    print(f"||f1 TCI - f1 tensor|| / ||f1 tensor|| = {rerr_f1}")   
    
    rerr_f2 = tl.norm(f2_tensor - recon_f2) / tl.norm(f2_tensor)
    print(f"||f2 TCI - f2 tensor|| / ||f2 tensor|| = {rerr_f2}")

    red_f1f2 = recon_f1 * recon_f2
    rerr_f1f2 = tl.norm(red_f1f2- g_tensor) / tl.norm(g_tensor)
    print(f"||f1 TCI x f2 TCI - f1 tensor x f2 tensor|| / ||f1 tensor x f2 tensor|| = {rerr_f1f2}")
    
    for r_max in range(10,13):
        _, TT_cores_f1f2, _, _, _, _, _ = TCI_2site(red_f1f2, 0, r_max, Nested_I_rank1, Nested_J_rank1)
        recon_f1f2 = tl.tt_to_tensor(TT_cores_f1f2)
        error = tl.norm(recon_f1f2 - g_tensor) / tl.norm(g_tensor)
        print(f"Relative error of g RED at r_max = {r_max}: {error}")    
    return

#red_test()

# Plot numerical results
max_rank_selection = [2,3,4,5,6,7,8]
rel_error_TCI_g = [0.518606738251041, 0.03929485422630780, 0.005080629328349756, 0.0022383885452628596, 0.0001335697037330801, 6.460937515610282e-05, 5.4462581444022203e-05]
rel_error_INT_g = [0.387611297071045, 0.03474862630575235, 0.007601452070678471, 0.0021256243069158304, 0.0002251037306136618, 2.463364284198022e-05, 1.963423284112322e-05]
max_rank_union = [4, 7]
rel_error_UNI_g = [0.016386594417341145, 0.0001748485620257831]

plt.figure()
#plt.scatter(max_rank_union, rel_error_UNI_g, label="Union-prrlu-f1f2", marker='o',color='orange')
plt.plot(max_rank_selection, rel_error_INT_g,color='green')
plt.scatter(max_rank_selection, rel_error_INT_g, label="QTTM", marker='s',color='green')
#plt.scatter(max_rank_selection, rel_error_TCI_g, label="TCI-prrlu-2site (RED)", marker='x',color='red')
plt.yscale("log")
plt.grid()
plt.legend()
plt.xlabel("Maximum TT-Rank")
plt.ylabel("Relative difference from the real f1xf2")
plt.savefig("relerr_UNI_vs_TCI_vs_INT.png")
plt.savefig('relerr_UNI_vs_TCI_vs_INT.svg', format='svg', bbox_inches='tight')
