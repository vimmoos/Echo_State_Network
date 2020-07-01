import numpy as np
from scipy import linalg
import project.esn.utils as u


@u.mydataclass(init=True, repr=True,frozen=True)
class Trainer():
    _trainer: callable
    ex_state: np.ndarray = np.zeros((0, 0))
    desired: np.ndarray = np.zeros((0, 0))
    param: float = 1e-8

    def __call__(self,other=None):
        ex_state,desired = (self.ex_state,self.desired) if other is None else other
        return self._trainer(ex_state,desired,self.param)


d_trainer = lambda fun: lambda *args, **kwargs: Trainer(fun, *args, **kwargs)


@d_trainer
@u.pre_proc_args({"desired": u.force_2dim})
def ridge_reg(ex_state: np.ndarray, desired: np.ndarray, reg_coef: float):
    tmp = linalg.inv(
        ex_state.T.dot(ex_state) + (np.eye(ex_state.shape[1]) * reg_coef))
    return tmp.dot(ex_state.T.dot(desired)).T
