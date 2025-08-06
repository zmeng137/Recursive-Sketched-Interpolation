import numpy as np
import tensorly as tl

from QTT import quantics_generation, adj_ttcore_contract
from tensor_cross import TCI_2site, Rank1_Nested_initIJ_gen, slice_first_modes, slice_last_modes, cross_core_interp_assemble, cross_inv_merge
from functions import Function_Collection

''' === Quantics representation construction === '''
# Quantics construction
func1 = Function_Collection[1]
dim = 12
x_tensor, f1_tensor = quantics_generation(func1, dim)

''' === Tensor cross interpolation for f1, f2, g === '''
# Create initial (rank-1) interpolation I/J sets
Nested_I_rank1, Nested_J_rank1 = Rank1_Nested_initIJ_gen(f1_tensor)

# TCI-2site of f1
r_max = 5
eps = 1e-8
TT_cross_f1, TT_cores_f1, TTRank_f1, pr_set_f1, pc_set_f1, interp_I_f1, interp_J_f1 = TCI_2site(f1_tensor, eps, r_max, Nested_I_rank1, Nested_J_rank1)

# Check the quality of QTCI
recon_f1 = tl.tt_to_tensor(TT_cores_f1)
rel_error = tl.norm(recon_f1 - f1_tensor) / tl.norm(f1_tensor)
print(f"Relative error of f1 QTT at r_max = {r_max}: {rel_error}")

offset = 4
core_1 = TT_cores_f1[offset]
core_2 = TT_cross_f1[2 * (offset + 1)]
contracted_core = adj_ttcore_contract(core_1, core_2)

I_set = interp_I_f1[offset]
J_set = interp_J_f1[offset + 3]
for i in range(len(I_set)):
    tensor_slice_i = slice_first_modes(f1_tensor, I_set[i].astype(int).tolist())
    for j in range(len(J_set)):
        tensor_slice_ij = slice_last_modes(tensor_slice_i, J_set[j].astype(int).tolist())
        for m in range(2):
            for n in range(2):         
                print(f"Value at I={I_set[i]}, Free Indices {m, n}, J={J_set[j]}: {tensor_slice_ij[m, n]}. Contraction approximation {contracted_core[i, m, n, j]}.")
    