import os
import sys
import numpy as np
import time as tm
import tensorly as tl
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'py'))
from tci import TT_IDPRRLDU_L2R
from map_rsi import NonlinearMapTT_RSI
from utils import size_tt, load_quantics_tensor_formula, Function_Collection, convert_quantics_tensor_to_1d

''' ===================== Tensor Generation ===================== '''

def quantics_function_tensor():
    # Load quantics function tensors from synthetic formulas 
    func_no = 11
    f = Function_Collection[func_no]
    qtensor_f, _ = load_quantics_tensor_formula(func_no, 20)

    # TT Decomposition of f
    r_max_f = 15
    eps = 0
    TTCore_f, TTRank_f, _ = TT_IDPRRLDU_L2R(qtensor_f, r_max_f, eps, 0)
    recon_f = tl.tt_to_tensor(TTCore_f)
    error_f = tl.norm(qtensor_f - recon_f) / tl.norm(qtensor_f)
    print(f"Relative error of TT_f (r_max = {r_max_f}): {error_f}")
    print(f"Size of full qtensor_f {qtensor_f.size}. Size of f1 QTT {size_tt(TTCore_f)}, QTT compression ratio {qtensor_f.size / size_tt(TTCore_f)}")
    return TTCore_f, qtensor_f, f

def general_synthetic_tensor():
    # Settings
    shape = [5, 5, 5, 5, 5, 5, 5, 5]
    ttrank = [1, 4, 11, 8, 10, 25, 8, 3, 1]
    seed = 10

    # Random TT
    random_tt = tl.random.random_tt(shape, ttrank, False, seed)
     
    return random_tt

print("Generating test tensor...")
#tt_f = general_synthetic_tensor()
tt_f, tensor_f, f_func = quantics_function_tensor()
print("Tensor generated.")

''' ===================== Functional TT Test ===================== '''

# Function g to be applied to the tensor network
#g_func = lambda x: np.cos(-2*x) * x + 2 - np.sin(3 * x) - x**3 / 100 + np.exp(-x*x/10)
#g_func = lambda x: x ** 2
#g_func = lambda x: np.maximum(0, x)
#g_func = lambda x: 1 / (1 + np.exp(-x))  # sigmoid function
g_func = lambda x: np.maximum(0,x)           # ReLU function
real_g = g_func(tensor_f)

# Recursive Sketching Interpolative Algorithm
ifEval = True
error_dict_rsi = {}
contract_number = [2]
r_max = [10,15,20,25,30,35]
oversampling = [20]
error_va_rank_rsi = []

for con in contract_number:
    for p in oversampling:
        for rm in r_max:
            seed = 10
            eps=0
            sketch_dim = int(rm/2) + p # oversampling = ...
            TTg_rsi, TTRank_rsi, _ = NonlinearMapTT_RSI(tt_f, g_func, con, rm, eps, sketch_dim, seed)

            if ifEval:
                g_rsi = tl.tt_to_tensor(TTg_rsi)
                err_g_rsi = np.linalg.norm(real_g - g_rsi) / np.linalg.norm(real_g)
                error_dict_rsi[f"cont_no={con}; rmax={rm}; oversampling={p}"] = err_g_rsi
                error_va_rank_rsi.append(err_g_rsi)

# Output errors
print("\n === RSI Method Errors ===\n")
for key, value in error_dict_rsi.items():
    print(f"{key}: {value}")
print("\n =========================\n")


# Plotting
ifPlot = True
if ifPlot:
    rsi_g_color = "#C6605C"
    true_g_color = "#6ECF8B"
    true_f_color = "#597ABA"
    
    # Create x values for plotting functions
    x = np.linspace(0, 1, 2**20)
    f_x = f_func(x)
    g_f_x = g_func(f_x)
    g_rsi = convert_quantics_tensor_to_1d(g_rsi)


    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # First subplot: f(x) and g(f(x))
    ax1.plot(x, f_x, label=r'$f(x)$', linewidth=2, color=true_f_color)
    ax1.plot(x, g_f_x, label=r'$g(f(x)) = \max(0, f(x))$', linewidth=4, color=true_g_color)
    ax1.plot(x, g_rsi, label=r'$g^{\text{RSI}}(\chi_{\max}(g^{\text{RSI}})=35)$', linewidth=4, color=rsi_g_color, linestyle=':')
    
    ax1.set_xlabel('x', fontsize=16)
    ax1.set_ylabel(r'(a) Function Value', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_xticklabels([0.0,0.2,0.4,0.6,0.8,1.0], fontsize=15)

    ax2.plot(r_max, error_va_rank_rsi, marker='o', linewidth=2, markersize=8)
    ax2.set_xlabel(r'$\chi_{\max}(g^{\text{RSI}})$', fontsize=16)
    ax2.set_ylabel(r'(b) Relative Error $\epsilon_r$', fontsize=16)
   
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='major')
    ax2.set_xticks(r_max)
    ax2.set_xticklabels(r_max,fontsize=15)


    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4)
    plt.savefig('function_shapes_and_error_convergence.pdf', dpi=300)