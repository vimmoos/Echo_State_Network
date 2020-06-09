
import numpy as np
from scipy import sparse,linalg

def generate_smatrix(m,n,density=0.5,bound=0.5):
    smatrix = sparse.rand(m,n,density=density,format="csr",random_state=42)
    smatrix[np.where(smatrix>0)] -= bound
    return smatrix

def scale_spectral_smatrix(matrix,spectral_radius=0.95,in_place=False):
    if not in_place:
        return matrix * (spectral_radius /
                         max(abs(linalg.eig(matrix)[0])))
    matrix *= (spectral_radius /
               max(abs(linalg.eig(matrix)[0])))

def one_step_update(W_in,W_res,state,input,leaking_rate=0.3,non_linearity=np.tanh):
    update = non_linearity(np.dot(W_in,input)
                           + np.dot(W_res,state))
    return (1-leaking_rate) * state  +  leaking_rate * update
