import numpy as np
import tensorly as tl

from tt_svd import TT_SVD
from tensor_cross import TT_CUR_L2R, TT_CUR_R2L, cross_core_interp_assemble, cross_inv_merge

# Tensor 
nl_func1 = lambda t: t ** 2 + 10 * np.ones(np.shape(t))
nl_func2 = lambda t: t ** 2
np.random.seed(50)
shape = [5, 5, 5, 5]
tensor = np.random.uniform(-10, 10, shape)
#tensor = nl_func1(tensor) 
nl_tensor = nl_func2(tensor)

r_max = 3
eps = 1e-8
ttList1, ttList_cc1, TTRank1, Nested_J = TT_CUR_R2L(tensor, r_max, eps)
ttList2, ttList_cc2, TTRank2, Nested_I = TT_CUR_L2R(tensor, r_max, eps)


new_cores = cross_inv_merge(ttList_cc1,4)
recon_new = tl.tt_to_tensor(new_cores)
error_new = tl.norm(tensor - recon_new) / tl.norm(tensor)
print(f"New recon error {error_new}.")

recon1 = tl.tt_to_tensor(ttList1)
recon2 = tl.tt_to_tensor(ttList2)
error1 = tl.norm(tensor - recon1) / tl.norm(tensor)
error2 = tl.norm(tensor - recon2) / tl.norm(tensor)    
print(f"TTCUR-R2L error {error1}. TTCUR-L2R error {error2}. ")


# PROBLEM!######
TTCore_nest_cross = cross_core_interp_assemble(tensor, Nested_I, Nested_J, [1,5,5,5,1])
new_cores = cross_inv_merge(TTCore_nest_cross,4)
recon_new = tl.tt_to_tensor(new_cores)
error_new = tl.norm(tensor - recon_new) / tl.norm(tensor)
print(f"New recon error {error_new}.")
#############TODO

ttList1 = TT_SVD(tensor, r_max, eps)
ttList2 = TT_SVD(nl_tensor, r_max, eps)
recon1 = tl.tt_to_tensor(ttList1)
recon2 = tl.tt_to_tensor(ttList2)
error1 = tl.norm(tensor - recon1) / tl.norm(tensor)
error2 = tl.norm(nl_tensor - recon2) / tl.norm(nl_tensor)    
print(f"TTSVD: error1 {error1}, error2 {error2}")

ttList1, ttList_cc1, TTRank, Nested_J = TT_CUR_L2R(tensor, r_max, eps)
ttList2, ttList_cc2, TTRank, Nested_J = TT_CUR_L2R(nl_tensor, r_max, eps)
recon1 = tl.tt_to_tensor(ttList1)
recon2 = tl.tt_to_tensor(ttList2)
error1 = tl.norm(tensor - recon1) / tl.norm(tensor)
error2 = tl.norm(nl_tensor - recon2) / tl.norm(nl_tensor)    
print(f"TTCUR: error1 {error1}, error2 {error2}")