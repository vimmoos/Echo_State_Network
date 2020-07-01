import numpy as np
from scipy import sparse, stats

def generate_smatrix(m, n, density=1, bound=0.5, **kwargs):
    return sparse.random(m,
                            n,
                            density=density,
                            format="csr",
                            data_rvs=stats.uniform(-bound, 1).rvs)



def scale_spectral_smatrix(matrix: sparse.issparse,
                                   spectral_radius=1.25,
                                   in_place=False):
    try:
        eigs = sparse.linalg.eigs(matrix)[0]
    except sparse.linalg.ArpackNoConvergence as e:
        eigs = e.eigenvalues
    finally:
        spectral = (spectral_radius /
                    max(abs(eigs)))

    return matrix * spectral


if __name__ == "__main__":
    from timeit import default_timer as timer
    from pprint import pprint
    tempos = [(timer(),scale_spectral_smatrix(generate_smatrix(2000,2000,0.0001)),timer()) for _ in range(1000)]
    stat = np.mean([z-x for (x,_,z) in tempos])
    pprint(stat)
