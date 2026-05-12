import numpy as np
from ncon import ncon
import npmps

I = np.array([[1.,0.],[0.,1.]])
sp = np.array([[0,1],[0,0]])
sm = np.array([[0,0],[1,0]])
su = np.array([[1.,0.],[0.,0.]])
sd = np.array([[0.,0.],[0.,1.]])
def make_tensorA_df2 ():
    A = np.zeros ((3,2,2,3)) # (k1,ipr,i,k2)
    A[0,:,:,0] = I
    A[1,:,:,0] = sp
    A[2,:,:,0] = sm
    A[1,:,:,1] = sm
    A[2,:,:,2] = sp
    return A

def make_LR_df2 ():
    L = np.array([-2.,1.,1.])
    R = np.array([1.,1.,1.])
    return L, R

def diff2_MPO (N, x1, x2):
    dx = (x2-x1)/(2**N)
    mpo = []
    for i in range(N):
        mpo.append (make_tensorA_df2())
    L, R = make_LR_df2()
    L *= 1./dx**2
    mpo = npmps.absort_LR (mpo, L, R)
    return mpo

def diff_MPO (N, x1, x2):
    dx = (x2-x1)/(2**N)
    mpo = []
    for i in range(N):
        mpo.append (make_tensorA_df2())

    L = np.array([0.,1.,-1.])
    R = np.array([1.,1.,1.])
    L *= 0.5/dx

    mpo = npmps.absort_LR (mpo, L, R)
    return mpo

def diff_MPO_not_antisymm (N, x1, x2):
    dx = (x2-x1)/(2**N)

    A = np.zeros ((2,2,2,2)) # (k1,ipr,i,k2)
    A[0,:,:,0] = I
    A[1,:,:,0] = sp
    A[1,:,:,1] = sm

    mpo = []
    for i in range(N):
        mpo.append (A)

    L = np.array([-1.,1.])
    R = np.array([1.,1.])
    L *= 1./dx

    mpo = npmps.absort_LR (mpo, L, R)
    return mpo
