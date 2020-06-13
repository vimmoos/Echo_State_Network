
import numpy as np
from scipy import linalg

def ridge_reg(ex_state,desired,reg_coef = 1e-8,**kwargs):
    tmp = linalg.inv(np.dot(ex_state.T,ex_state) +
                     reg_coef*np.eye(ex_state.shape[1]))
    return np.dot(tmp,
                  np.dot(ex_state.T,desired)).T
