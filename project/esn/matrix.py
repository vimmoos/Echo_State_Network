import numpy as np
from scipy import sparse,stats,linalg

def generate_smatrix(m,n,density=0.5,bound=0.5):
    np.random.seed(42)
    smatrix = sparse.random(m,n,density=density,
                            format="csr",
                            data_rvs = stats.uniform(-bound,1).rvs)
    return smatrix

def scale_spectral_smatrix(matrix,spectral_radius=1.25,in_place=False):
    if not in_place:
        return matrix * (spectral_radius /
                         max(abs(sparse.linalg.eigs(matrix)[0])))
    matrix *= (spectral_radius /
               max(abs(sparse.linalg.eigs(matrix)[0])))
