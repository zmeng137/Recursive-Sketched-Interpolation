import numpy as np
import sparse as sp
import time as tm
import matplotlib.pyplot as plt

# Visualization of 2D matrix 
def view_mat2d(matrix):
    plt.figure(figsize=(10,8))
    plt.imshow(matrix, cmap='viridis', aspect='auto')
    plt.tight_layout()
    plt.show()

# Show sparsity information of the input full (dense-format) matrix
def MatrixSparseStat(matrix: np.array):
    print("The sparsity statistics of the input matrix is as follows ...")
    size = matrix.size
    shape = matrix.shape
    cntzero = np.count_nonzero(np.abs(matrix) < 1e-10)
    cntnzero = size - cntzero
    sparsity = cntzero / size
    density = cntnzero / size
    print(f"shape = {shape}, size = {size}, # zero = {cntzero}, sparsity = {sparsity}, # non-zero = {cntnzero}, density = {density}")
    return

# Show sparsity information of the input full (dense-format) tensor array 
def TensorSparseStat(factors: list[np.array]):
    print("The sparsity statistics of the input tensor is as follows ...")
    totalnnz = 0
    for i in range(len(factors)):
        factor = factors[i]
        size = factor.size
        shape = factor.shape
        cntzero = np.count_nonzero(np.abs(factor) < 1e-10)
        cntnzero = size - cntzero
        totalnnz += cntnzero
        sparsity = cntzero / size
        density = cntnzero / size
        print(f"Tensor factor {i}: shape = {shape}, size = {size}, # zero = {cntzero}, sparsity = {sparsity}, # non-zero = {cntnzero}, density = {density}")
    print(f"Total number of non-zero: {totalnnz}\n\n")
    return

# Find Pct%-close-to-0 values of a martrix and cast them to 0
def CastValueAroundZero(Mat: np.array, Pct: float) -> np.array:
    absMat = np.abs(Mat)
    eleCnt = Mat.size
    # Sort and cast
    sortIndx = np.unravel_index(np.argsort(absMat, axis=None), absMat.shape)
    sortMat = absMat[sortIndx]
    if Pct >= 1.0 or Pct <= 0.0:
        print("The input percentage should be between 0 and 1. The percentage is set to 0.5 by default.")
        Pct = 0.5
    thresIdx = int(eleCnt * Pct) 
    thresVal = sortMat[thresIdx]    
    Mat = np.where(absMat > thresVal, Mat, 0)
    return Mat

# Read FROSTT data
def readfrostt(path: str, shape: tuple):
    print(f"Start loading tensor data {path}...")
    start_t = tm.time()
    #is_it_equal_function = lambda x:x in shape
    shape = np.array(shape)
    with open(path, "r") as file:
        modeList = []
        for line in file:
            numbers = line.strip().split()
            numbers = np.array([int(num) for num in numbers])
            coords = numbers[:-1]-1
            cmp = coords < shape
            if np.all(cmp):
                modeList.append(numbers)
        modeList = np.array(modeList, dtype=np.int32)
        modeList = modeList.T
        data = modeList[-1] 
        coords = modeList[:-1]-1
        spt = sp.COO(coords, data, shape)
    end_t = tm.time()
    print(f"Finish loading! It took {end_t - start_t} seconds.")
    return spt

def read_from_tns(filename, shape):
    data_array = np.loadtxt(filename) # Load all data using numpy (fast method)
    
    # Split into coordinates and values
    coords = data_array[:, :-1].astype(int)  # All columns except last as coordinates
    values = data_array[:, -1]  # Last column as values
    
    # Create COO array
    return sp.COO(coords.T, values, shape=shape)