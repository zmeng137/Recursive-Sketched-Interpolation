import os
import h5py
import itertools
import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt
from datetime import datetime

# Synthetic function collection
Function_Collection = {}

# Normal functions
Function_Collection[0] = lambda x: 1
Function_Collection[1] = lambda x: 1.2 * x ** 4 - 0.2 * np.sqrt(x) - 1 + 0.6 * np.sin(7.3 * np.pi * x)  
Function_Collection[2] = lambda x: -1.1 * x ** 7 - 12 + np.exp(3.1 * x) - 0.81 * np.cos(6 * np.pi * x) - 2 * x ** 2 + 4 + np.tan(x) 
Function_Collection[3] = lambda x: 10 * np.exp(- (x - 1) * (x - 1) / (2 * 0.15 * 0.15))
Function_Collection[4] = lambda x: 100 * np.exp(- (x - 0) * (x - 0) / (2 * 0.3 * 0.3))

# High oscillation functions
B_const = 7
Function_Collection[5] = lambda x: np.cos(x * (2 ** B_const)) * np.cos(x * (2 ** B_const) / (4 * np.sqrt(5))) * np.exp(- x * x) + 2 * np.exp(-x)
Function_Collection[6] = lambda x: np.sin(x * (2 ** B_const)) / (2 * np.sqrt(10)) * np.exp(- x * x * x) - 4 * np.exp(-np.sqrt(x))

# 1D Gaussian functions
w = 0.15
w1,w2 = w,w
x1 = 0.45
x2 = 0.55
Function_Collection[7] = lambda x: np.exp(-(x-x1)**2/w1**2)
Function_Collection[8] = lambda x: np.exp(-(x-x2)**2/w2**2)

# Save quantics tensor to HDF5 file with metadata.
def save_quantics_tensor_hdf5(quantics_tensor, filepath, metadata=None, compression=True):
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    with h5py.File(filepath, 'w') as f:
        # Save the quantics tensor data
        if compression:
            # Use gzip compression with chunk size optimized for tensor structure
            chunk_size = tuple(min(dim, 64) for dim in quantics_tensor.shape)
            dataset = f.create_dataset(
                'quantics_tensor',
                data=quantics_tensor,
                compression='gzip',
                compression_opts=6,  # Compression level 0-9
                chunks=chunk_size,
                shuffle=True,        # Reorder bytes for better compression
                fletcher32=True      # Add checksum for data integrity
            )
        else:
            dataset = f.create_dataset('quantics_tensor', data=quantics_tensor)
        
        # Store tensor attributes
        dataset.attrs['shape'] = quantics_tensor.shape
        dataset.attrs['dtype'] = str(quantics_tensor.dtype)
        dataset.attrs['n_modes'] = len(quantics_tensor.shape)
        dataset.attrs['total_elements'] = quantics_tensor.size
        dataset.attrs['memory_mb'] = quantics_tensor.nbytes / (1024**2)
        
        # Store creation timestamp
        dataset.attrs['created_at'] = datetime.now().isoformat()
        dataset.attrs['format_version'] = '1.0'
        dataset.attrs['description'] = 'Quantics tensor in hierarchical binary format'
        
        # Store optional metadata
        if metadata is not None:
            metadata_group = f.create_group('metadata')
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata_group.attrs[key] = value
                elif isinstance(value, np.ndarray):
                    metadata_group.create_dataset(key, data=value)
                elif isinstance(value, (list, tuple)):
                    metadata_group.attrs[key] = np.array(value)
                else:
                    # Convert to string for unsupported types
                    metadata_group.attrs[key] = str(value)
        
        # Calculate and store compression statistics
        if compression:
            uncompressed_size = quantics_tensor.nbytes
            compressed_size = os.path.getsize(filepath)
            compression_ratio = uncompressed_size / compressed_size
            
            f.attrs['uncompressed_size_mb'] = uncompressed_size / (1024**2)
            f.attrs['compressed_size_mb'] = compressed_size / (1024**2)
            f.attrs['compression_ratio'] = compression_ratio
    
    print(f"Quantics tensor saved to: {filepath}")
    if compression:
        print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"File size: {os.path.getsize(filepath) / (1024**2):.2f} MB")

# Load quantics tensor from HDF5 file.
def load_quantics_tensor_hdf5(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with h5py.File(filepath, 'r') as f:
        # Load the quantics tensor
        quantics_tensor = f['quantics_tensor'][:]
        
        # Load attributes
        attrs = dict(f['quantics_tensor'].attrs)
        
        # Load metadata if exists
        metadata = {}
        if 'metadata' in f:
            metadata_group = f['metadata']
            
            # Load metadata attributes
            for key, value in metadata_group.attrs.items():
                metadata[key] = value
            
            # Load metadata datasets
            for key in metadata_group.keys():
                metadata[key] = metadata_group[key][:]
        
        # Load file-level attributes
        file_attrs = dict(f.attrs)
        
        # Combine all metadata
        full_metadata = {
            'tensor_attrs': attrs,
            'file_attrs': file_attrs,
            'user_metadata': metadata
        }
    
    print(f"Quantics tensor loaded from: {filepath}")
    print(f"Shape: {quantics_tensor.shape}")
    print(f"Dtype: {quantics_tensor.dtype}")
    print(f"Size: {quantics_tensor.nbytes / (1024**2):.2f} MB")
    
    return quantics_tensor, full_metadata

# Convert 1D function to quantics tensor format recursively
def convert_1d_to_quantics_tensor(function, n_bits):
    if len(function) != 2**n_bits:
        raise ValueError(f"Function length {len(function)} != 2^{n_bits} = {2**n_bits}")
    
    if n_bits == 1:
        #quantics_tensor = torch.zeros(2,dtype=function.dtype)
        quantics_tensor = np.zeros(2,dtype=np.float32)
        quantics_tensor[0] = function[0]
        quantics_tensor[1] = function[1]
        return quantics_tensor
    
    # Create the quantics tensor
    quantics_shape = (2,) * n_bits
    #quantics_tensor = torch.zeros(quantics_shape, dtype=function.dtype)
    quantics_tensor = np.zeros(quantics_shape, dtype=np.float32)
    
    new_bit = n_bits - 1
    func_half_1 = function[0 : 2 ** new_bit]
    func_half_2 = function[2 ** new_bit :] 

    quantics_tensor[0,:] = convert_1d_to_quantics_tensor(func_half_1, new_bit)
    quantics_tensor[1,:] = convert_1d_to_quantics_tensor(func_half_2, new_bit)
    
    return quantics_tensor

# Convert quantics tensor format to 1D function array recursively
def convert_quantics_tensor_to_1d(quantics_tensor):
    if quantics_tensor.size == 2:
        return np.array([quantics_tensor[0], quantics_tensor[1]])
    
    l1 = quantics_tensor.size
    l2 = l1 >> 1

    function_1D = np.zeros(l1)
    quantics_tensor_0 = quantics_tensor[0,:]
    quantics_tensor_1 = quantics_tensor[1,:]

    function_1D[0:l2] = convert_quantics_tensor_to_1d(quantics_tensor_0)
    function_1D[l2:]  = convert_quantics_tensor_to_1d(quantics_tensor_1)

    return function_1D

# Compute the total size of a tensor train
def size_tt(TTCores):
    size = 0
    for core in TTCores:
        size += core.size
    
    return size

# Populate tensor using numpy.fromfunction
def populate_tensor_fromfunction(dims, func):
    # Populate tensor using numpy.fromfunction
    def array_func(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12):
        return func(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)    
    
    # Use fromfunction to create the tensor
    tensor_data = np.fromfunction(array_func, dims, dtype=int)
    
    return tl.tensor(tensor_data)

# Scatter plot for f1, f2, g
def scatter_plot_f1f2(x_tensor, g_tensor, f1_tensor = None, f2_tensor = None):
    plt.figure()
    plt.scatter(x_tensor, g_tensor, s=8, alpha=0.8, linewidth=0.5, label='g')
    
    if f1_tensor is not None:
        plt.scatter(x_tensor, f1_tensor, s=4, alpha=0.8, linewidth=0.5, label='f1')
    if f2_tensor is not None:
        plt.scatter(x_tensor, f2_tensor, s=4, alpha=0.8, linewidth=0.5, label='f2')
    
    plt.legend()
    plt.grid()
    plt.savefig("f1_f2_g.png")
    
    return

# Generate quantics representation of a continuous function
def quantics_generation(func, digit):
    print("Generating quantics tensor...")
    shape = tuple([2] * digit)   # Tensor shape 2^n
    x_tensor = np.zeros(shape)   # Quantics x tensor
    f_tensor = np.zeros(shape)   # Quantics function 
    
    # Generate all possible combinations of indices (0,1) for n dimensions
    for indices in itertools.product([0, 1], repeat = digit):
        # Calculate the value using the formula: x1/2 + x2/2^2 + ... + xn/2^n
        value = sum(x / (2 ** (i + 1)) for i, x in enumerate(indices))
        x_tensor[indices] = value
        f_tensor[indices] = func(value)
    print("Quantics tensor generation completed.")

    return x_tensor, f_tensor

def quantics_generation_fast(func, digit):
    print("Generating quantics tensor...")
    
    # Create a meshgrid for all binary combinations
    grids = np.meshgrid(*[np.array([0, 1])] * digit, indexing='ij')
    
    # Stack grids and compute values vectorized
    indices_array = np.stack(grids, axis=-1)  # shape: (2,2,...,2,digit)
    
    # Vectorized calculation: x1/2 + x2/2^2 + ... + xn/2^n
    powers = 2.0 ** np.arange(1, digit + 1)
    x_tensor = np.sum(indices_array / powers, axis=-1)
    
    # Apply function (vectorized if possible, otherwise use np.vectorize)
    try:
        f_tensor = func(x_tensor)
    except:
        f_tensor = np.vectorize(func)(x_tensor)
    
    print("Quantics tensor generation completed.")
    return x_tensor, f_tensor

# Construct a synthetic quantics tensor from given formula
def load_quantics_tensor_formula(form_no, dim):
    func1 = Function_Collection[form_no]
    x_tensor, f1_tensor = quantics_generation_fast(func1, dim)
    
    return f1_tensor, x_tensor

def tt_to_tensor_tensordot(TT):
    # Tensor train -> Reconstructed tensor
    tt_cores = TT.copy()    
    result = tt_cores[0]
    for i in range(1, len(tt_cores)):
        result = np.tensordot(result, tt_cores[i], axes=([result.ndim-1], [0]))
    result = np.squeeze(result, axis=-1)
    
    return result

# Contract two adjacent TT-cores
def adj_ttcore_contract(core1, core2):
    if core1.shape[2] != core2.shape[0]:
        raise ValueError(f"Incompatible shapes: core1 rank {core1.shape[2]} != core2 rank {core2.shape[0]}")
    
    # Perform the contraction via einsum
    contracted = np.einsum('air,rjb->aijb', core1, core2)
    
    return contracted