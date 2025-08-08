import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt

from qtt import populate_tensor_fromfunction, union_rows_bounded, scatter_plot_f1f2, integral_qtt
from tensor_cross import TT_CUR_L2R, cross_core_interp_assemble, TCI_2site, cross_inv_merge

# Quantics construction
quantic_repres = lambda x1,x2,x3,x4,x5,x6,x7,x8,x9,x10: x1/2 + x2/(2**2) + x3/(2**3) + x4/(2**4) + x5/(2**5) + x6/(2**6) + x7/(2**7) + x8/(2**8) + x9/(2**9) + x10/(2**10)
func = lambda t: np.abs(2.2 * t ** 6.3 + 2.2 * np.sqrt(t) - 1 + 0.6 * np.sin(8.3 * np.pi * t))  #t ** 5 - 3 * t ** 3 + 10 * t -6 #5 * np.sin(-2 * np.pi * t) - 3 * np.exp(t)
g_func = lambda t: func(t) * func(t)
shape = (2,2,2,2,2,2,2,2,2,2)
dim = len(shape)
x_tensor = populate_tensor_fromfunction(shape, quantic_repres)
f_tensor = func(x_tensor)
g_tensor = f_tensor * f_tensor
scatter_plot_f1f2(x_tensor, g_tensor, f_tensor)

''' === Generate initial pivots === '''
TTCores, TTCores_Cross, TTRank, Nested_I_rank1, Nested_J_rank1 = TT_CUR_L2R(f_tensor, 1, 0)
Assemble_TTCore_Cross = cross_core_interp_assemble(f_tensor, Nested_I_rank1, Nested_J_rank1, TTRank)
for i in range(2 * len(TTRank) - 3):
    diff_flag = (Assemble_TTCore_Cross[i] == TTCores_Cross[i]).all()
    if (diff_flag == False):
        print(f"Interpolation assembly error at {i}!")
Nested_I_rank1[0] = []
Nested_J_rank1[dim+1] = []

''' === TCI on different ranks === '''
err_QTCI_f_fi = []
err_QTCI_g_gi = []
err_QTCI_g_fi = []
r_max_config = [1,2,3,4,5,6,7,8]
for r_max in r_max_config:
    print(f"\nMaximum TT-Rank at {r_max}:")
    # 1. TCI-2site of f(t)
    interp_I_f, interp_J_f, TTRank_f, recon_f = TCI_2site(f_tensor, 1e-12, r_max, Nested_I_rank1, Nested_J_rank1)
    error = tl.norm(f_tensor - recon_f) / tl.norm(f_tensor)
    err_QTCI_f_fi.append(error)
    print(f"|| QTCI_f (using f interp) - f || / || f || at r_max = {r_max}: {error}")    
    
    # 2. TCI-2site of g(t) = f(t) * f(t)
    interp_I_g, interp_J_g, TTRank_g, recon_g = TCI_2site(g_tensor, 1e-12, r_max, Nested_I_rank1, Nested_J_rank1)
    error = tl.norm(g_tensor - recon_g) / tl.norm(g_tensor)
    err_QTCI_g_gi.append(error)
    print(f"|| QTCI_g (using g interp) - g || / || g || at r_max = {r_max}: {error}")
    
    # 3. Assemble g(t) by f's interpolation pivots
    TT_cross = cross_core_interp_assemble(g_tensor, interp_I_f, interp_J_f, TTRank_f)    
    TT_cores = cross_inv_merge(TT_cross, dim)
    recon_g_Interpf = tl.tt_to_tensor(TT_cores)
    error = tl.norm(g_tensor - recon_g_Interpf) / tl.norm(g_tensor)
    err_QTCI_g_fi.append(error)
    print(f"|| QTCI_g (using f interp) - g || / || g || at r_max = {r_max}: {error}")

plt.figure()
plt.plot(r_max_config, err_QTCI_f_fi, label='QTCI of f (via optimal f interpolation)')
plt.plot(r_max_config, err_QTCI_g_gi, label='QTCI of g (via optimal g interpolation)')
plt.plot(r_max_config, err_QTCI_g_fi, label='QTCI of g (via optimal f interpolation)')
plt.legend()
plt.grid()
plt.savefig("square_error.png")