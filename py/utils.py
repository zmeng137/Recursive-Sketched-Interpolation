import numpy as np
import h5py
import os
from datetime import datetime

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