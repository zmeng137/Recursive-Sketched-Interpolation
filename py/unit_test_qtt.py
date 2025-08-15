import numpy as np
import tensorly as tl

from qtt import quantics_generation, QTT_Sketching, integral_qtt
from utils import Function_Collection

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MLA-Toolkit', 'py'))
from tensor_cross import TT_CUR_L2R

def unit_test_1():
    '''Unit test for sketching of quantics tensor trains'''
    # Initialization
    dim = 12
    func = Function_Collection[1]
    x_tensor, f_tensor = quantics_generation(func, dim)
    
    # TT-ID 
    r_max = 12
    eps = 1e-15
    TTCores, _, TTRank, _, _ = TT_CUR_L2R(f_tensor, r_max, eps, 0, 0)

    # Reconstruction 
    recon = tl.tt_to_tensor(TTCores)
    error = tl.norm(f_tensor - recon) / tl.norm(f_tensor)
    print(f"Relative error of QTT at r_max = {r_max}: {error}")
    print(f"The QTT-rank is {TTRank}")

    # Integral sketching 
    skdim = 5
    SkQTT_1 = QTT_Sketching(TTCores, skdim, False, 0, 0)

    # Random multi-layer sketching
    skdim = 5
    SkQTT_2 = QTT_Sketching(TTCores, skdim, True, 0, 3)

    # TEST: Whether the un-sketched parts are same or not 
    for i in range(dim - skdim - 1):
        unsk_diff = tl.norm(SkQTT_1[i] -SkQTT_2[i])
        if unsk_diff != 0:
            print("ERROR! Un-sketched QTTs are different.")
    
    # TEST: Integral 
    full_int_bench = np.sum(f_tensor) / 2**dim
    full_int_1 = integral_qtt(TTCores, dim)
    full_int_2 = QTT_Sketching(TTCores, dim-1, False, 0, 0)
    full_int_2 = np.squeeze(full_int_2[0])
    full_int_2 = 0.5 * full_int_2[0] + 0.5 * full_int_2[1]
    print(f"Real best integral: {full_int_bench}; QTT integral {full_int_1, full_int_2}")   

    pass


unit_test_1()