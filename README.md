# Recursive Sketched Interpolation (RSI)
### Efficient Hadamard Products of Tensor Trains

This repository contains the implementation of the Recursive Sketched Interpolation (RSI) algorithm introduced in the paper:

> **Recursive Sketched Interpolation: Efficient Hadamard Products of Tensor Trains**
> Zhaonan Meng, Yuehaw Khoo, Jiajia Li, E. Miles Stoudenmire
> [arXiv:2602.17974](https://arxiv.org/abs/2602.17974)

---

## Overview

Computing the **Hadamard (element-wise) product** of two tensor trains (TTs) is a fundamental operation in tensor network methods. The naive approach yields a product TT whose bond dimension grows as the product of the input bond dimensions, and recompressing it via SVD-based rounding costs at least **O(χ⁴)** in the bond dimension χ.

This work introduces **RSI (Recursive Sketched Interpolation)**, which combines:
- **Randomized TT sketching** — compresses the trailing modes of the TT using SRHT or similar random maps, avoiding explicit formation of the full product tensor.
- **Interpolative decomposition (ID)** — selects representative rows/columns to form compressed TT cores, maintaining a nested interpolation structure across sites.

RSI reduces the complexity to **O(χ³)** while achieving accuracy comparable to traditional methods. The algorithm generalizes naturally to Hadamard products of **multiple TTs** and other **nonlinear element-wise mappings** of a TT, g(TT), without increasing complexity beyond O(χ³).

---

## Dependencies

Install the required packages with pip:

```bash
pip install numpy scipy tensorly h5py matplotlib
```

---

## Usage

All algorithm implementations are located in `/py`. and all test scripts in `test/` add `../py` to `sys.path` automatically, so no package installation is required beyond the dependencies above.

### Hadamard product of two TTs (RSI)

```python
import sys
sys.path.append('py/')
from multiply_rsi import HadamardTT_RSI

# tt1, tt2: lists of 3D numpy arrays with shape (bond_dim_left, n, bond_dim_right)
tt_g, ranks_g, interp_sets = HadamardTT_RSI(
    tt1, tt2,
    contract_core_number=2,  # number of indices to be open (unsketched)
    max_rank=100,            # maximum bond dimension of output TT
    eps=0,                   # ID truncation tolerance (0 = use max_rank only)
    sketch_dim=50,           # sketching dimension
    seed=1                   # random seed for randmized sketching
)
```

### Hadamard product of multiple TTs

```python
from multiply_rsi import HadamardTT_RSI_fs

TTset = [tt1, tt2, tt3]   # list of TTs to multiply element-wise
tt_g, ranks_g, interp_sets = HadamardTT_RSI_fs(
    TTset, contract_core_number=2, max_rank=100, eps=0, sketch_dim=50, seed=1
)
```

### Nonlinear mapping g(f(x))

```python
from map_rsi import NonlinearMapTT_RSI

g_func = lambda x: np.maximum(x, 0)   # e.g., ReLU
tt_g, ranks_g = NonlinearMapTT_RSI(tt_f, g_func, max_rank=50, eps=1e-6, sketch_dim=50, seed=1)
```

---

## Experiments

### DMRG MPS benchmark (`test/mps_dmrg.py`)

Computes the Hadamard product of a DMRG ground-state MPS with itself (i.e., |ψ|²) and evaluates accuracy via:
- Relative error
- Diagonal energy deviation

Compares RSI vs. direct Kronecker + SVD rounding on systems with different sites and bond dimensions.

```bash
python test/mps_dmrg.py
```

### Quantics TT (`test/quantics_tt.py`)

Test quantics tensor train representations of functions (such as Gaussian functions).

```bash
python test/quantics_tt.py
```

### Nonlinear mapping (`test/nonlinear_map.py`)

Applies a nonlinear function (e.g., ReLU, polynomial) to a quantics TT encoding a synthetic 1D function, and measures the relative reconstruction error against the exact result.

```bash
python test/nonlinear_map.py
```

---

## Data

Datasets are stored in HDF5 format under `datasets/`. Pre-generated quantics tensors and MPS files can be reproduced using the scripts in `py/data_generation/`.

| Dataset | Description |
|---------|-------------|
| `itensor_dmrg_mps/` | DMRG ground-state MPS for 1D spin chains (n = 10, 15, 20, 50 sites, generated with ITensor) |
| `qtensor_function/` | Quantics TT encodings of synthetic 1D/multidimensional functions |
| `qtensor_well/` | Quantics TT encodings of PDE solutions from the [Well dataset](https://github.com/PolymathicAI/the_well) (active matter) |

---
