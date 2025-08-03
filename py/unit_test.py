import numpy as np

import tensorly as tl
from tensorly.random import random_tt
from tensorly.tt_tensor import tt_to_tensor

from utils import generate_smooth_svd_matrix
from rank_revealing import prrldu, PivotedQR
from interpolation import interpolative_qr, interpolative_nuclear, interpolative_prrldu, cur_prrldu
from tensor_cross import TT_CUR

def unit_test_1():
    # Test of interpolative_nuclear
    print("Unit test 1 starts!")
    A = np.array([[3,1],[8,2],[9,-5],[-7,4]])
    B = np.array([[4,6,2],[8,-1,-4]])    
    M = A @ B
    
    maxdim = 2
    cutoff = 1e-4
    C, X, cols, error = interpolative_nuclear(M, cutoff, maxdim)
    error = np.linalg.norm(M - C @ X, ord='fro') / np.linalg.norm(M, ord='fro')    
    #print(f"M - C*X=\n{M - C @ X}")
    
    ut_statement = "Test succeeds!" if error < cutoff else "Test fails!"
    print(f"relative error={error}, " + ut_statement)
    print("Unit test 1 ends!")
    return

def unit_test_3():
    # Test of interpolative_prrldu
    print("Unit test 3 starts!")
    m = 12
    n = 11
    rank = 8
    A = np.random.random((m, rank))
    B = np.random.random((rank, n))
    M = A @ B

    cutoff = 1E-5
    maxdim = 9
    C, Z, pivot_cols, inf_error = interpolative_prrldu(M, cutoff, 9)
    error = np.linalg.norm(M - C @ Z, ord='fro') / np.linalg.norm(M, ord='fro')    
    #print(f"M - C*X=\n{M - C @ X}")
    
    ut_statement = "Test succeeds!" if error < cutoff else "Test fails!"
    print(f"relative error={error}, " + ut_statement)
    print("Unit test 3 ends!")
    return

def unit_test_4():
    print("Unit test 4 starts!")
    m = 250
    r = 150
    n = 300
    M = np.random.random((m,r)) @ np.random.random((r,n))
    cutoff = 1E-10

    approx, C, Z = interpolative_qr(M, 150)
    error = np.linalg.norm(M - approx, ord='fro') / np.linalg.norm(M, ord='fro')    
    
    ut_statement = "Test succeeds!" if error < cutoff else "Test fails!"
    print(f"relative error={error}, " + ut_statement)
    print("Unit test 4 ends!")
    return

def prrldu_test():
    print("Unit test of partial rank-revealing LDU factorization starts!")
    # Random rank-deficient test matrix
    m = 50
    n = 40
    rank = 30
    min_val = 1
    max_val = 100
    A = np.random.uniform(min_val, max_val, (m,rank))
    B = np.random.uniform(min_val, max_val, (rank,n))
    M = A @ B

    cutoff = 1e-8
    maxdim = 50
    mindim = 1    
    L, d, U, row_perm_inv, col_perm_inv, inf_error = prrldu(M, cutoff, maxdim, mindim)
    
    recon = L @ np.diag(d) @ U
    recon_recover_r = recon[row_perm_inv,:]
    recon_recover_rc = recon_recover_r[:,col_perm_inv]
    max_err = np.max(np.abs(recon_recover_rc - M))    
    print(f"prrldu: revealed rank = {L.shape[1]}, max error = {max_err}")    
    print("Unit test ends!")
    return

def cur_test():
    print("CUR Decomposition test 1:")
    A = np.array([[3,1],[8,2],[9,-5],[-7,4]])
    B = np.array([[4,6,2],[8,-1,-4]])    
    M = A @ B
    maxdim = 2
    cutoff = 1e-4
    r_subset, c_subset, cross_inv, cross, rank = cur_prrldu(M, cutoff, maxdim)
    error = np.linalg.norm(M - c_subset @ cross_inv @ r_subset, ord='fro') / np.linalg.norm(M, ord='fro')    
    ut_statement = "Test succeeds!" if error < cutoff else "Test fails!"
    print(f"relative error={error}, " + ut_statement)
    
    print("CUR Decomposition test 2:")
    m = 50
    n = 40
    rank = 30
    min_val = 1
    max_val = 100
    A = np.random.uniform(min_val, max_val, (m,rank))
    B = np.random.uniform(min_val, max_val, (rank,n))
    M = A @ B
    cutoff = 1e-8
    maxdim = 50
    r_subset, c_subset, cross_inv, cross, rank = cur_prrldu(M, cutoff, maxdim)
    error = np.linalg.norm(M - c_subset @ cross_inv @ r_subset, ord='fro') / np.linalg.norm(M, ord='fro')
    ut_statement = "Test succeeds!" if error < cutoff else "Test fails!"
    print(f"relative error={error}, " + ut_statement)
    return

def elesquare_cur_test():
    m = 8
    n = 6
    rank = 4
    min_val = -10
    max_val = 10
    A = np.random.uniform(min_val, max_val, (m,rank))
    B = np.random.uniform(min_val, max_val, (rank,n))
    M = A @ B
    print(f"M max: {np.max(M)}") 
    print(f"M min: {np.min(M)}")
    cutoff = 1e-8
    maxdim = 100
    r_subset, c_subset, cross_inv, cross, rank = cur_prrldu(M, cutoff, maxdim)
    L, d, U, row_perm_inv, col_perm_inv, rps, cps, inf_error = prrldu(cross, cutoff, rank)
    
    cross_elesq = cross ** 2
    L, d, U, row_perm_inv, col_perm_inv, rps, cps, inf_error = prrldu(cross_elesq, cutoff, rank)
    

    M_elesq = M ** 2
    r_subset, c_subset, cross_inv, cross, rank = cur_prrldu(M_elesq, cutoff, maxdim)
    L, d, U, row_perm_inv, col_perm_inv, rps, cps, inf_error = prrldu(cross, cutoff, rank)

    r_elesq = r_subset ** 2
    c_elesq = c_subset ** 2
    cross_inv_elesq = np.linalg.inv(cross_elesq)
    error = np.linalg.norm(M_elesq - c_elesq @ cross_inv_elesq @ r_elesq, ord='fro') / np.linalg.norm(M_elesq, ord='fro')
    ut_statement = "Test succeeds!" if error < cutoff else "Test fails!"
    print(f"relative error={error}, " + ut_statement)
    
    return

def tt_cur_test():
    print("TT-CUR Decomposition test 1:")
    shape = [10, 10, 10, 10]
    rank = [1, 4, 20, 8, 1]
    tensor = random_tt(shape, rank, full=True)
    r_max = max(rank)
    eps = 1e-8
    ttList, ttList_cc = TT_CUR(tensor, r_max, eps)
    recon = tt_to_tensor(ttList)
    error = tl.norm(tensor - recon) / tl.norm(tensor)
    ut_statement = "Test succeeds!" if error < eps else "Test fails!"
    print(f"relative error={error}, " + ut_statement)

    print("TT-CUR Decomposition test 2:")
    shape = [30, 51, 21, 4]
    rank = [1, 43, 60, 3, 1]
    tensor = random_tt(shape, rank, full=True)
    r_max = max(rank)
    eps = 1e-8
    ttList, ttList_cc = TT_CUR(tensor, r_max, eps)
    recon = tt_to_tensor(ttList)
    error = tl.norm(tensor - recon) / tl.norm(tensor)
    ut_statement = "Test succeeds!" if error < eps else "Test fails!"
    print(f"relative error={error}, " + ut_statement)
    return


shape = [3, 2, 4, 6]
rank = [1, 2, 5, 3, 1]
tensor = random_tt(shape, rank, full=True)
r_max = 3
eps = 1e-8
ttList, ttList_cc = TT_CUR(tensor, r_max, eps)
recon = tt_to_tensor(ttList)
error = tl.norm(tensor - recon) / tl.norm(tensor)
print(f"relative error={error}")

M = generate_smooth_svd_matrix(20, 20, 0.3)
cutoff = 1e-8

print("====== Original matrix M ======")
U, S, Vh = np.linalg.svd(M ** 1)
print(f"Singular values: {S}")
maxdim = 7
r_subset, c_subset, cross_inv, cross, rank, row_sample, col_sample = cur_prrldu(M, cutoff, maxdim)
error = np.linalg.norm(M - c_subset @ cross_inv @ r_subset, ord='fro') / np.linalg.norm(M, ord='fro')
print(f"Relative error: {error}")
print(f"Row pivoting {row_sample}")
print(f"Col pivoting {col_sample}")

r_subset_nl = r_subset ** 2
c_subset_nl = c_subset ** 2
cross_nl_inv = np.linalg.inv(cross ** 2)
error = np.linalg.norm(M ** 2 - c_subset_nl @ cross_nl_inv @ r_subset_nl, ord='fro') / np.linalg.norm(M ** 2, ord='fro')
print(f"Relative error: {error}")

maxdim = 8
nl_recon = (c_subset @ cross_inv @ r_subset) ** 2
r_subset, c_subset, cross_inv, cross, rank, row_sample, col_sample = cur_prrldu(nl_recon, cutoff, maxdim)
error = np.linalg.norm(M ** 2 - c_subset @ cross_inv @ r_subset, ord='fro') / np.linalg.norm(M ** 2, ord='fro')
print(f"Relative error: {error}")
print(f"Row pivoting {row_sample}")
print(f"Col pivoting {col_sample}")



print("====== M ** 2 ======")
U, S, Vh = np.linalg.svd(M ** 2)
print(f"Singular values: {S}")
maxdim = 7
r_subset, c_subset, cross_inv, cross, rank, row_sample, col_sample = cur_prrldu(M**2, cutoff, maxdim)
error = np.linalg.norm(M**2 - c_subset @ cross_inv @ r_subset, ord='fro') / np.linalg.norm(M ** 2, ord='fro')
print(f"Relative error: {error}")
print(f"Row pivoting {row_sample}")
print(f"Col pivoting {col_sample}")

print("====== M ** 4 ======")
U, S, Vh = np.linalg.svd(M ** 4)
print(f"Singular values: {S}")
maxdim = 7
r_subset, c_subset, cross_inv, cross, rank, row_sample, col_sample = cur_prrldu(M**4, cutoff, maxdim)
error = np.linalg.norm(M**4 - c_subset @ cross_inv @ r_subset, ord='fro') / np.linalg.norm(M ** 4, ord='fro')
print(f"Relative error: {error}")
print(f"Row pivoting {row_sample}")
print(f"Col pivoting {col_sample}")

pass
