# RSI QTT Density Workflow

This project computes a 2D QTT/MPS density (|ψ|² of a rotated-GPE wavefunction) using two methods:

- `HadamardTT_RSI` in `gpe_density.py` — RSI algorithm ([arXiv:2602.17974](https://arxiv.org/abs/2602.17974))
- `npmps.SRC` in `gpe_density.py` — Successive Randomized Compression baseline ([arXiv:2504.06475](https://arxiv.org/abs/2504.06475))

RSI is extended here to support an MPO acting on an MPS and to support the complex data type, so the density is built as `conj(psi)`-as-MPO times `psi`.

The plotting script `plot2D_sec.py` then converts one saved MPS back to a 2D grid and saves the center-section plot, the 2D density map, and the complex grid data.

## Required Packages

Use Python 3.10+ (tested on 3.12).

Install the packages used by the current scripts and the shared `../src` modules:

```bash
pip install numpy scipy matplotlib ncon tensorly h5py jax jaxlib numba
```

Notes:

- `matplotlib` is configured with `plt.rc("text", usetex=True)`, so PDF plotting may require a local LaTeX installation. If LaTeX is not installed, set `usetex=False` in `plot2D_sec.py`.
- `jax`/`jaxlib` can be CPU-only unless you specifically want GPU acceleration.

## Files

- `91v.pkl`: input wavefunction MPS, it describe the rotated GPE and contain 91 vortices.
- `GPE paper`: (https://arxiv.org/abs/2507.04279)
- `gpe_density.py`: builds density-like MPS objects from the input wavefunction.
- `plot2D_sec.py`: plots a saved density MPS on a 2D grid.
- MPS/MPO/QTT helper code lives in the repository's `src/` directory (`npmps`, `qtt`, `differential`, `plot_utility_jax`), which the scripts add to `sys.path` via `../src`.

## How To Run

Run commands from the `gpe_density/` directory (`gpe_density.py` writes its outputs to the current working directory):

```bash
cd gpe_density
python gpe_density.py
```

This takes ~20 s and prints the two density-MPS norms (RSI vs. SRC should agree) and saves:

- `psi2_hada.pkl`: density MPS from `HadamardTT_RSI`
- `psi2_src.pkl`: density MPS from `npmps.SRC`

Before plotting, set `INPUT_MPS` near the top of `plot2D_sec.py` to the file you want to visualize. Note that the checked-in default points at the committed sample `density_hadamard_rsi_mps.pkl` (same content as a `psi2_hada.pkl` run); to plot a file you just generated, change it to:

```python
INPUT_MPS = SCRIPT_DIR / "psi2_hada.pkl"
```

or:

```python
INPUT_MPS = SCRIPT_DIR / "psi2_src.pkl"
```

Then run:

```bash
python plot2D_sec.py
```

This evaluates the MPS on the full grid (~10⁶ points after coarse-graining) via the JAX/numba batched evaluator, so it takes a few minutes. The plotting script saves:

- `<input_stem>_coarse_mps.pkl`: coarse-grained MPS after `kill_site_2D`
- `<input_stem>_grid_complex.tsv`: real and imaginary grid values
- `<input_stem>_center_section.pdf`: center-section plot at `y = 0`
- `<input_stem>_2d_map.pdf`: 2D density map

## Main Parameters

### `gpe_density.py`

- `N = 13`: number of QTT binary sites per spatial dimension.
- `MAX_RANK = 80`: maximum bond dimension for the RSI/SRC output.
- `INPUT_MPS`: input wavefunction MPS pickle file.
- `contract_core_number=N`: number of open/unsketched cores used by `HadamardTT_RSI`.
- `eps=0`: no tolerance truncation; rank is controlled by `MAX_RANK`.
- `sketch_dim=1`: randomized sketch dimension.
- `seed=1`: random seed for reproducibility.

### `plot2D_sec.py`

- `INPUT_MPS`: density MPS file to plot.
- `X1, X2 = -21, 21`: physical coordinate range.
- `KILL_SITE_TIMES = 3`: number of coarse-graining passes.
- `MAX_RANK = 80`: rank parameter passed to `kill_site_2D`.
- `BATCH_SIZE = 8192 * 3`: batch size for grid evaluation.

The grid size is inferred from the MPS:

```python
n_sites = len(mps) // 2
n_grid = 2**n_sites
dx = (X2 - X1) / n_grid
```

## QTT Ordering

For a 2D system, the MPS stores all `x` QTT bits first, then all `y` QTT bits:

```text
[x_0, x_1, ..., x_{N-1}, y_{N-1}, y_{N-2}, ..., y_{0}]
```

Here `N = len(mps) // 2`.

`plot2D_sec.py` uses the same binary list for `x` and `y`:

```python
binary_numbers = list(pltut.BinaryNumbers(n_sites))
xs = pltut.bin_to_dec_list(binary_numbers, scale, X1)
x_mesh, y_mesh = np.meshgrid(xs, xs)
```

The physical coordinate mapping is:

```text
coordinate = binary_integer * dx + X1
dx = (X2 - X1) / 2**N
```

## MPS Leg Definition

Every MPS tensor is a rank-3 tensor with legs:

```text
(left_bond, physical_index, right_bond)
```

So each site tensor has shape:

```text
(chi_left, d, chi_right)
```

For this QTT code, the local physical dimension is usually:

```text
d = 2
```

Boundary conditions:

```text
mps[0].shape[0] == 1
mps[-1].shape[-1] == 1
```

The helper `npmps.check_MPS_links(mps)` verifies that neighboring virtual bonds match:

```text
mps[i - 1].shape[-1] == mps[i].shape[0]
```

## MPO Leg Definition

Every MPO tensor is a rank-4 tensor with legs:

```text
(left_bond, output_physical_index, input_physical_index, right_bond)
```

So each MPO site tensor has shape:

```text
(chi_left, d_out, d_in, chi_right)
```

For square local operators in this code:

```text
d_out == d_in == 2
```

Boundary conditions:

```text
mpo[0].shape[0] == 1
mpo[-1].shape[-1] == 1
```

The helper `npmps.check_MPO_links(mpo)` verifies that neighboring virtual bonds match and that all local physical dimensions are consistent.

## MPS To MPO Conversion

`qtt.MPS_to_MPO(mps)` converts each MPS tensor into a diagonal MPO tensor:

```text
MPS tensor: (left, i, right)
MPO tensor: (left, i_out, i_in, right)
```

For each left/right virtual-bond pair, the physical vector is placed on the diagonal:

```text
T[left, :, :, right] = diag(A[left, :, right])
```

This is used in `gpe_density.py` to build an operator from `conj(psi)` before multiplying it with `psi`.

## Normalization Notes

`gpe_density.py` loads the input MPS and applies:

```python
npmps.normalize_MPS(mps)
```

This normalizes using the discrete MPS inner product.

`qtt.normalize_MPS_by_integral(mps, x1, x2, Dim=2)` normalizes with grid-spacing factors for continuous-space interpretation. The plotting script currently rescales the first tensor by `1 / dx**2` before coarse-graining.

## Troubleshooting

- `ModuleNotFoundError: No module named 'ncon'`: install `ncon`.
- LaTeX errors while saving PDF: install LaTeX or set `plt.rc("text", usetex=False)`.
- Missing input file in `plot2D_sec.py`: make sure `INPUT_MPS` matches one of the files saved by `gpe_density.py`.
