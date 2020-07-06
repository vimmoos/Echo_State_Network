import numpy as np
from scipy import sparse, stats, linalg
from project.esn.utils import mydataclass, pre_proc_args, force_2dim
import pickle as pic
''' generator for scipy and np matrix '''


def generate_smatrix(m, n, density=1, bound=0.5, **kwargs):
    smatrix = sparse.random(m,
                            n,
                            density=density,
                            format="csr",
                            data_rvs=stats.uniform(-bound, 1).rvs)
    return smatrix


def generate_rmatrix(m, n, bound=0.5, **kwargs):
    return np.random.rand(m, n) - bound


def scale_spectral_smatrix(matrix: sparse.issparse,
                                   spectral_radius=1.25,
                                   in_place=False):
    eigs = None
    try:
        eigs = sparse.linalg.eigs(matrix)[0]
    except sparse.linalg.ArpackNoConvergence as e:
        eigs = e.eigenvalues
    finally:
        spectral = (spectral_radius / max(abs(eigs)))
    if not in_place:
        return matrix * spectral
    matrix *= spectral


from pprint import pprint


@pre_proc_args({"inputs": force_2dim, "states": force_2dim})
def build_extended_states(inputs: np.ndarray, states: np.ndarray, init_len=0):
    return np.vstack((inputs.T[:, init_len:], states.T[:, init_len:])).T


@mydataclass(init=True, repr=True, check=False)
class Esn_matrixs():
    W_in: np.ndarray
    W_res: sparse.issparse
    W_feb: np.ndarray = np.zeros((0, 0))
    W_out: np.ndarray = np.zeros((0, 0))
    spectral_radius: float = 1.25
    density: float = 1.

    def __post_init__(self):
        scale_spectral_smatrix(self.W_res,
                               spectral_radius=self.spectral_radius,
                               in_place=True)


esn_matrixs = lambda W_in, *args, **kwargs: Esn_matrixs(
    W_in, generate_smatrix(W_in.shape[0], W_in.shape[0], **kwargs), *args, **
    kwargs)

def load_smatrix(path, idx):
    with open(path, "rb") as f:
        dic = pic.load(f)
        return {
            k if k != "result" else "W_res": v if k != "result" else v[idx]
            for k, v in dic.items() if not k in ["repetition", "size"]
        }

read_matrix = lambda W_in, path, idx, *args, **kwargs: Esn_matrixs(
    W_in, *args, **load_smatrix(path, idx), **kwargs)
