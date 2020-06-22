import numpy as np
from scipy import sparse, stats, linalg
from project.esn import utils as u
import project.esn.matrix as m


def apply_leak(state, update, leaking_rate=0.3):
    return (1 - leaking_rate) * state + leaking_rate * update


@u.mydataclass(repr=True, init=True, check=True)
class Updator():
    _next: callable
    weights: m.Esn_matrixs
    squeeze_f: callable
    state: np.ndarray
    leaking_rate: float = 1
    noise: float = 0

    def __post_init__(self):
        self._next = u.comp(self.squeeze_f, lambda x: x + self.noise,
                            self._next)

    def __lshift__(self, other):
        next_state = self._next(self.weights, self.state, other)
        self.state = apply_leak(self.state, next_state, self.leaking_rate)
        return next_state


def_updator = lambda fun: lambda *args, **kwargs: Updator(fun, *args, **kwargs)


@def_updator
def vanilla_updator(weights: m.Esn_matrixs, state, tuple):
    (input, _) = tuple
    return (weights.W_in * input) + (weights.W_res * state)


@def_updator
def feedback_updator(weights: m.Esn_matrixs, state, tuple):
    (input, output) = tuple
    return ((weights.W_in * input) + (weights.W_res * state) +
            (weights.W_feb * output))


''' TODO rewrite better !!!
'''


def default_output(W_out, input, state, output_f=lambda x: x):
    z_n = u.build_extended_states(np.matrix(input), np.matrix(state)).T
    return output_f(W_out * z_n).reshape(-1)
