import numpy as np
from scipy import sparse, stats, linalg
from project.esn.utils import mydataclass, pre_proc_args, force_2dim
import pickle as pic
from dataclasses import field

""" generator for scipy and np matrix """


def generate_smatrix(m, n, density=1, bound=0.5, **kwargs):
    """generate a sparse matrix in the CSR format (Compressed Sparse Row)"""
    smatrix = sparse.random(
        m, n, density=density, format="csr", data_rvs=stats.uniform(-bound, 1).rvs
    )
    return smatrix


def generate_rmatrix(m, n, bound=0.5, **kwargs):
    """generate a random dense matrix"""
    return np.random.rand(m, n) - bound


def scale_spectral_smatrix(
    matrix: sparse.issparse, spectral_radius=1.25, in_place=False
):
    """calculate the spectral radius and scale the matrix"""
    eigs = None
    try:
        eigs = sparse.linalg.eigs(matrix)[0]
    except sparse.linalg.ArpackNoConvergence as e:
        eigs = e.eigenvalues
    finally:
        spectral = spectral_radius / max(abs(eigs))
    if not in_place:
        return matrix * spectral
    matrix *= spectral


from pprint import pprint


@pre_proc_args({"inputs": force_2dim, "states": force_2dim})
def build_extended_states(inputs: np.ndarray, states: np.ndarray, init_len=0):
    """create the extend state given an input array and a state array"""
    return np.vstack((inputs.T[:, init_len:], states.T[:, init_len:])).T


@mydataclass(init=True, repr=True, check=False)
class Esn_matrixs:
    """Wraps all the matrixs needed by the network"""

    W_in: np.ndarray
    W_res: sparse.issparse
    W_feb: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    W_out: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    spectral_radius: float = 1.25
    density: float = 1.0
    scaled: bool = False

    def __post_init__(self):
        if not self.scaled:
            scale_spectral_smatrix(
                self.W_res, spectral_radius=self.spectral_radius, in_place=True
            )


esn_matrixs = lambda W_in, *args, **kwargs: Esn_matrixs(
    W_in, generate_smatrix(W_in.shape[0], W_in.shape[0], **kwargs), *args, **kwargs
)


def load_smatrix(path, idx):
    """used to load a matrix from a pickled file"""
    with open(path, "rb") as f:
        dic = pic.load(f)
        return {
            k if k != "result" else "W_res": v if k != "result" else v[idx]
            for k, v in dic.items()
            if not k in ["repetition", "size"]
        }


read_matrix = lambda W_in, path, idx, *args, **kwargs: Esn_matrixs(
    W_in, *args, **load_smatrix(path, idx), **kwargs
)
