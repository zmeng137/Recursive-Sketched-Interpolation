import copy
import os
import pickle
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"

ROOT_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT_DIR / "NumpyTensorTools"
sys.path.append(str(TOOLS_DIR))

import numpy as np
import npmps
import qtt_tools as qtt
from multiply_rsi import HadamardTT_RSI


N = 13
MAX_RANK = 80
# Input the .pkl file name for the MPS of the wavefunction. The output will be saved as "psi2_hada.pkl" and "psi2_src.pkl" for HadamardTT_RSI and SRC, respectively.
INPUT_MPS = Path(__file__).with_name("91v.pkl")
OUTPUTS = {
    "hada": Path("psi2_hada.pkl"),
    "src": Path("psi2_src.pkl"),
}


def load_mps(path):
    with open(path, "rb") as file:
        return npmps.normalize_MPS(pickle.load(file))


def save_pickle(path, data):
    with open(path, "wb") as file:
        pickle.dump(data, file)


def build_psi_squared(psi):
    psi_op = npmps.conj(qtt.MPS_to_MPO(psi))

    psi2_hada = HadamardTT_RSI(
        psi_op,
        psi,
        contract_core_number=N,
        max_rank=MAX_RANK,
        eps=0,
        sketch_dim=1,
        seed=1,
        verbose=False,
    )
    psi2_src = npmps.SRC(psi_op, psi, MAX_RANK)
    return psi2_hada, psi2_src


def print_norm(label, mps):
    norm = np.sqrt(npmps.inner_MPS(mps, mps))
    print(f"{label}_norm:", norm)


def main():
    psi = copy.copy(load_mps(INPUT_MPS))
    psi2_hada, psi2_src = build_psi_squared(psi)

    print_norm("hadama", psi2_hada)
    print_norm("src", psi2_src)
    print(psi[12].shape)

    save_pickle(OUTPUTS["hada"], psi2_hada)
    save_pickle(OUTPUTS["src"], psi2_src)


if __name__ == "__main__":
    main()
