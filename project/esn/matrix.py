import numpy as np
from scipy import sparse,stats,linalg

def generate_smatrix(m,n,density=0.5,bound=0.5,**kwargs):
    np.random.seed(42)
    smatrix = sparse.random(m,n,density=density,
                            format="csr",
                            data_rvs = stats.uniform(-bound,1).rvs)
    return smatrix

def scale_spectral_smatrix(matrix,spectral_radius=0.9,in_place=False,**kwargs):
    if not in_place:
        return matrix * (spectral_radius /
                         max(abs(sparse.linalg.eigs(matrix)[0])))
    matrix *= (spectral_radius /
               max(abs(sparse.linalg.eigs(matrix)[0])))

def generate_rresvoir(m,density=0.5,bound=0.5,spectral_radius=0.9,**kwargs):
    return scale_spectral_smatrix(
        generate_smatrix(m,m,density,bound),
        spectral_radius)

def generate_rmatrix(m,n,bound=0.5,**kwargs):
    return np.random.rand(m,n) - bound
