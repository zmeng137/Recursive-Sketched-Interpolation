import numpy as np
import tensorly as tl
from interpolation import interpolative_prrldu, cur_prrldu, cur_prrldu_ninv
from tensor_cross import TT_CUR

# 3D Input (3D tensor)
np.random.seed(1)
shape = [50, 50, 50]
tensor = np.random.uniform(-5, 5, shape)
nl_func = lambda t: t ** 2 + 0 * np.ones(np.shape(t))
nl_tensor = nl_func(tensor)

W = tensor
nbar = W.size  # Total size of W
r = 1               # Rank r
r_max = 40
eps = 1e-8
TT_cores = []

ttList1, ttList_cc1 = TT_CUR(tensor, r_max, eps)
ttList2, ttList_cc2 = TT_CUR(nl_tensor, r_max, eps)
recon1 = tl.tt_to_tensor(ttList1)
recon2 = tl.tt_to_tensor(ttList2)
error1 = tl.norm(tensor - recon1) / tl.norm(tensor)
error2 = tl.norm(nl_tensor - recon2) / tl.norm(nl_tensor)
print(f"TTCUR: error1 {error1}, error2 {error2}")



# Detail Procedure
curr_dim = shape[0]  # Current dimension
W = tl.reshape(W, [int(r * curr_dim), int(nbar / r / curr_dim)])  # Reshape W       
        
r_subset, c_subset, cross_inv, cross, rank, pr, pc = cur_prrldu(W, eps, r_max)
pr = pr[0:rank]  # Row skeleton 
pc = pc[0:rank]  # Col skeleton

Ti = tl.reshape(c_subset @ cross_inv, [r, shape[0], rank])
TT_cores.append(Ti)                                          
TT_cores.append(tl.reshape(c_subset, [r, shape[0], rank]))  
TT_cores.append(cross)
         
nbar = int(nbar * rank / shape[0] / r)  # New total size of W
r = rank  # Renewal r
W = r_subset[0:rank,:]     


