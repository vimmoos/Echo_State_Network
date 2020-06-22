import numpy as np
from scipy import linalg
import project.esn.utils as u
import project.esn.matrix as m


def ridge_reg(ex_state: Matrix, desired, reg_coef=1e-8):
    tmp = ~((ex_state.T() * ex_state) + (m.eye(ex_state.shape[1])**preg_coef))
    return (tmp * (ex_state.T() * desired)).T()
