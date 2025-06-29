import numpy as np
import tensorly as tl

from tt_svd import TT_SVD
from tensor_cross import TT_CUR_L2R, TT_CUR_R2L, cross_core_interp_assemble, cross_inv_merge

# Tensor
nl_func1 = lambda t: t ** 2 
nl_func2 = lambda t: np.exp(t) #+ 10 * np.ones(np.shape(t))
nl_func3 = lambda t: np.exp(t) * (t**2)
np.random.seed(50)
shape = [5, 5, 5, 5, 5]
tensor = np.random.uniform(-10, 10, shape)
#tensor = np.random.normal(0, 1, shape)
nl_tensor1 = nl_func1(tensor)
nl_tensor2 = nl_func2(tensor)
nl_tensor3 = nl_func3(tensor)

r_max = 21
eps = 1e-8

ttList1, ttList_cc1, TTRank1, Nested_J = TT_CUR_R2L(tensor, r_max, eps)
ttList2, ttList_cc2, TTRank2, Nested_I = TT_CUR_L2R(tensor, r_max, eps)
recon1 = tl.tt_to_tensor(ttList1)
recon2 = tl.tt_to_tensor(ttList2)
error1 = tl.norm(tensor - recon1) / tl.norm(tensor)
error2 = tl.norm(tensor - recon2) / tl.norm(tensor)    
print(f"TTCUR-R2L error {error1}. TTCUR-L2R error {error2}. ")

r_max = 10
eps = 1e-8

nl1_ttList1, nl1_ttList_cc1, nl1_TTRank1, nl1_Nested_J = TT_CUR_R2L(nl_tensor1, r_max, eps)
nl1_ttList2, nl1_ttList_cc2, nl1_TTRank2, nl1_Nested_I = TT_CUR_L2R(nl_tensor1, r_max, eps)
nl1_recon1 = tl.tt_to_tensor(nl1_ttList1)
nl1_recon2 = tl.tt_to_tensor(nl1_ttList2)
nl1_error1 = tl.norm(nl_tensor1 - nl1_recon1) / tl.norm(nl_tensor1)
nl1_error2 = tl.norm(nl_tensor1 - nl1_recon2) / tl.norm(nl_tensor1)    
print(f"TTCUR-R2L error {nl1_error1}. TTCUR-L2R error {nl1_error2}. ")

r_max = 10
eps = 1e-8

nl2_ttList1, nl2_ttList_cc1, nl2_TTRank1, nl2_Nested_J = TT_CUR_R2L(nl_tensor2, r_max, eps)
nl2_ttList2, nl2_ttList_cc2, nl2_TTRank2, nl2_Nested_I = TT_CUR_L2R(nl_tensor2, r_max, eps)
nl2_recon1 = tl.tt_to_tensor(nl2_ttList1)
nl2_recon2 = tl.tt_to_tensor(nl2_ttList2)
nl2_error1 = tl.norm(nl_tensor2 - nl2_recon1) / tl.norm(nl_tensor2)
nl2_error2 = tl.norm(nl_tensor2 - nl2_recon2) / tl.norm(nl_tensor2)    
print(f"TTCUR-R2L error {nl2_error1}. TTCUR-L2R error {nl2_error2}. ")

r_max = 10
eps = 1e-8

nl3_ttList1, nl3_ttList_cc1, nl3_TTRank1, nl3_Nested_J = TT_CUR_R2L(nl_tensor3, r_max, eps)
nl3_ttList2, nl3_ttList_cc2, nl3_TTRank2, nl3_Nested_I = TT_CUR_L2R(nl_tensor3, r_max, eps)
nl3_recon1 = tl.tt_to_tensor(nl3_ttList1)
nl3_recon2 = tl.tt_to_tensor(nl3_ttList2)
nl3_error1 = tl.norm(nl_tensor3 - nl3_recon1) / tl.norm(nl_tensor3)
nl3_error2 = tl.norm(nl_tensor3 - nl3_recon2) / tl.norm(nl_tensor3)    
print(f"TTCUR-R2L error {nl3_error1}. TTCUR-L2R error {nl3_error2}. ")

pass


#new_cores = cross_inv_merge(ttList_cc1,4)
#recon_new = tl.tt_to_tensor(new_cores)
#error_new = tl.norm(tensor - recon_new) / tl.norm(tensor)
#print(f"New recon error {error_new}.")



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