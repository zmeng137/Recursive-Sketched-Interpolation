import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt

from tt_svd import TT_SVD

# Let's say we have a 5 order quantics

# numpy.fromfunction
def populate_tensor_fromfunction(dims, func):
    # Populate tensor using numpy.fromfunction
    def array_func(x1, x2, x3, x4, x5):
        return func(x1, x2, x3, x4, x5)    
    # Use fromfunction to create the tensor
    tensor_data = np.fromfunction(array_func, dims, dtype=int)
    return tl.tensor(tensor_data)

quantic_repres = lambda x1,x2,x3,x4,x5: x1/2 + x2/(2**2) + x3/(2**3) + x4/(2**4) + x5/(2**5)
nonlinear_func = lambda t: t ** 2

tl.set_backend("numpy")
shape = (2,2,2,2,2)
x_tensor = populate_tensor_fromfunction(shape, quantic_repres)
f_tensor = nonlinear_func(x_tensor)

r_max = 3
eps = 1e-8
TTCores = TT_SVD(f_tensor, r_max, eps)
recon = tl.tt_to_tensor(TTCores)
error = tl.norm(f_tensor - recon) / tl.norm(f_tensor)

scatter = plt.scatter(x_tensor, f_tensor, 
                     c=f_tensor,  # Color by f values
                     cmap='viridis',  # Color map
                     s=60,          # Point size
                     alpha=0.7,  # Transparency
                     edgecolors='black',  # Point borders
                     linewidth=0.5)
plt.savefig("quantics.png")