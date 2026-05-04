import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, ScalarFormatter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../NumpyTensorTools")))

import npmps
import plot_utility_jax as pltut
import plotsetting as ps
import qtt_tools as qtt

plt.rc("text", usetex=True)
plt.rc("font", family="serif")


SCRIPT_DIR = Path(__file__).resolve().parent
#INPUT_MPS = SCRIPT_DIR / "density_src_mps.pkl" or "density_hadamard_rsi_mps.pkl" --- IGNORE ---
INPUT_MPS = SCRIPT_DIR / "density_hadamard_rsi_mps.pkl"
OUTPUT_STEM = INPUT_MPS.stem.removesuffix("_mps")
COMPLEX_DATA_PATH = SCRIPT_DIR / f"{OUTPUT_STEM}_grid_complex.tsv"
SECTION_PLOT_PATH = SCRIPT_DIR / f"{OUTPUT_STEM}_center_section.pdf"
DENSITY_MAP_PATH = SCRIPT_DIR / f"{OUTPUT_STEM}_2d_map.pdf"
X1, X2 = -21, 21

KILL_SITE_TIMES = 3
MAX_RANK = 80
BATCH_SIZE = 8192 * 3


def load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def save_pickle(path, data):
    with open(path, "wb") as file:
        pickle.dump(data, file)


def prepare_mps(path):
    mps = load_pickle(path)
    n_sites = len(mps) // 2
    n_grid = 2**n_sites
    scale = (X2 - X1) / n_grid
    mps[0] = mps[0]/scale**2
    for _ in range(KILL_SITE_TIMES):
        mps = qtt.kill_site_2D(mps, MAX_RANK, dtype=np.complex128)

    save_pickle(path.with_name(f"{path.stem}_coarse_mps.pkl"), mps)
    return mps

def build_mesh(mps):
    n_sites = len(mps) // 2
    n_grid = 2**n_sites
    scale = (X2 - X1) / n_grid

    binary_numbers = list(pltut.BinaryNumbers(n_sites))
    xs = pltut.bin_to_dec_list(binary_numbers, scale, X1)
    x_mesh, y_mesh = np.meshgrid(xs, xs)
    z_mesh = pltut.get_2D_mesh_eles_mps(
        mps,
        binary_numbers,
        binary_numbers,
        batch_size=BATCH_SIZE,
    )
    return x_mesh, y_mesh, z_mesh


def save_complex_data(z_mesh):
    data = np.column_stack((np.real(z_mesh).ravel(), np.imag(z_mesh).ravel()))
    np.savetxt(COMPLEX_DATA_PATH, data, fmt="%.10e", delimiter="\t")


def plot_center_section(x_mesh, density):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    center = density.shape[0] // 2
    x_values = x_mesh[center]
    section = density[center]
    ymax = float(np.nanmax(section))

    ax.plot(x_values, section, color="black", linewidth=1.8)
    ax.set_xlabel(r"$x$", fontsize=18)
    ax.set_ylabel(r"$|\psi(x, 0)|^{2}$", fontsize=18)
    ax.set_xlim(X1, X2)
    ax.set_ylim(bottom=0, top=ymax * 1.08 if ymax > 0 else 1)
    ax.xaxis.set_major_locator(MaxNLocator(7))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    y_formatter = ScalarFormatter(useMathText=True)
    y_formatter.set_powerlimits((-2, 3))
    ax.yaxis.set_major_formatter(y_formatter)
    ax.tick_params(axis="both", which="major", labelsize=14, length=5, width=1)
    ax.tick_params(axis="both", which="minor", length=3, width=0.8)
    ax.grid(True, which="major", color="0.86", linewidth=0.8)
    ax.grid(True, which="minor", color="0.93", linewidth=0.5)
    ps.text(ax, x=0.08, y=0.92, t="(a)", fontsize=16)

    fig.savefig(SECTION_PLOT_PATH, bbox_inches="tight", transparent=False)
    plt.close(fig)


def plot_density_map(x_mesh, y_mesh, density, output_path):
    fig, ax = plt.subplots()
    contour = ax.contourf(x_mesh, y_mesh, density, cmap=cm.coolwarm, antialiased=False)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(r"$x$", rotation=0, fontsize=20)
    ax.set_ylabel(r"$y$", rotation=0, fontsize=20)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(X1, X2)
    ax.set_ylim(X1, X2)
    ps.set_tick_inteval(ax.xaxis, major_itv=5, minor_itv=1)
    ps.set_tick_inteval(ax.yaxis, major_itv=5, minor_itv=1)
    ps.text(ax, x=0.1, y=0.9, t="(b)", fontsize=20)

    colorbar = fig.colorbar(contour)
    colorbar.ax.tick_params(labelsize=20)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    mps = prepare_mps(INPUT_MPS)
    print(mps[1])
    print(npmps.MPS_dims(mps))

    x_mesh, y_mesh, z_mesh = build_mesh(mps)
    save_complex_data(z_mesh)

    density = np.abs(z_mesh)
    plot_center_section(x_mesh, density)
    plot_density_map(x_mesh, y_mesh, density, DENSITY_MAP_PATH)


if __name__ == "__main__":
    main()
