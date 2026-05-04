import numpy as np
import copy
import npmps
import differential as df
from ncon import ncon
def grow_site_2D_0th (psi,dtype = np.complex128):
    assert len(psi) % 2 == 0
    psi = copy.copy(psi)
    t0 = np.array([[[1.],[1.]]],dtype=dtype)
    psi[0:0] = [t0]
    psi[len(psi):len(psi)] = [t0]
    npmps.check_MPS_links(psi)
    return psi

def grow_site_2D_1th (psi,maxdim,dtype = np.complex128):
    assert len(psi) % 2 == 0
    nsite = len(psi)
    psi = copy.copy(psi)
    psi_0th = grow_site_2D_0th(psi)
    psi_0th = copy.copy(psi_0th)
    bulk_x_mpo = np.zeros ((2,2,2,2),dtype=dtype)
    bulk_x_mpo[0,:,:,0] = df.I
    bulk_x_mpo[1,:,:,0] = df.sp
    bulk_x_mpo[1,:,:,1] = df.sm

    bulk_y_mpo = np.zeros ((2,2,2,2),dtype=dtype)
    bulk_y_mpo[0,:,:,0] = df.I
    bulk_y_mpo[1,:,:,0] = df.sm
    bulk_y_mpo[1,:,:,1] = df.sp

    left_xbond_tenop = np.zeros ((1,2,2,2),dtype=dtype)
    left_xbond_tenop[0,:,:,0] = df.su+0.5*df.sd
    left_xbond_tenop[0,:,:,1] = 0.5*df.sd    
    right_ybond_tenop = bulk_y_mpo[:,:,:,0:1]

    op = [left_xbond_tenop]+[bulk_x_mpo]*(nsite//2) + [bulk_y_mpo]*(nsite//2) + [right_ybond_tenop]
    psi_1th = []
    nsite_ext = len(psi_0th)
    for i in range(nsite_ext):
        ten = ncon ([op[i],psi_0th[i]], ((-1,-3,3,-4), (-2,3,-5)))
        arr = np.shape(ten)
        ten = ten.reshape((arr[0]*arr[1], arr[2], arr[3]*arr[4]))
        psi_1th[i:i] = [ten] 
    npmps.check_MPS_links(psi_1th)
    psi_1th = npmps.compress_MPS (psi_1th, maxdim=maxdim)
    return psi_1th

def kill_site_2D(psi, maxdim,dtype = np.complex128):
    assert len(psi) % 2 == 0
    nsite = len(psi)
    kill_ten = np.array([1/2,1/2],dtype=dtype)
    psi = copy.copy(psi)
    left_ten= ncon ([kill_ten,psi[0]], ((1,), (-1,1,-2)))
    right_ten= ncon ([kill_ten,psi[-1]], ((1,), (-1,1,-2)))
    kill_psi = [psi[i] for i in range(1,int(len(psi)-1))]
    kill_psi[0] = ncon ([left_ten,kill_psi[0]], ((-1,1), (1,-2,-3)))
    kill_psi[-1]=ncon ([right_ten,kill_psi[-1]], ((1,-3), (-1,-2,1)))
    npmps.check_MPS_links(kill_psi)
    #kill_psi = npmps.compress_MPS (kill_psi, maxdim=maxdim)
    return kill_psi


def MPS_tensor_to_MPO_tensor (A):
    assert A.ndim == 3
    T = np.zeros((A.shape[0],A.shape[1],A.shape[1],A.shape[2]), dtype=A.dtype)
    for i in range(A.shape[0]):
        for j in range(A.shape[2]):
            ele = A[i,:,j]
            T[i,:,:,j] = np.diag(ele)
    return T

def MPS_to_MPO (mps):
    npmps.check_MPS_links (mps)

    mpo = []
    for A in mps:
        T = MPS_tensor_to_MPO_tensor (A)
        mpo.append(T)
    return mpo

def normalize_MPS_by_integral (mps, x1, x2, Dim):
    mps = copy.copy(mps)
    c = npmps.inner_MPS (mps, mps)
    mps[0] = mps[0] / c**0.5

    N = len(mps)//Dim
    Ndx = 2**N
    dx = (x2-x1)/Ndx

    for d in range(Dim):
        i = d*N
        mps[i] = mps[i] / dx**0.5
    return mps

def sum_elements (mps):
    A = np.array([1,1]).reshape((1,2,1))
    mps2 = [A for i in range(len(mps))]
    return npmps.inner_MPS (mps, mps2)
