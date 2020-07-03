from pprint import pprint as p

import numpy as np
import project.esn.matrix as m
from project.esn import utils as u


def apply_leak(state, update, leaking_rate=0.3):
    return (update * leaking_rate) + (state * (1 - leaking_rate))


@u.mydataclass(repr=True, init=True, check=False)
class Updator():
    _next: callable
    weights: m.Esn_matrixs
    state: np.ndarray
    squeeze_f: callable = np.tanh
    squeeze_o : callable = lambda x : x
    leaking_rate: float = 0.3
    noise: float = 0.0

    def __post_init__(self):
        self._next = u.comp(self.squeeze_f, lambda x: x + self.noise,
                            self._next)

    def __call__(self, other):
        next_state = self._next(self.weights, self.state, other)
        self.state = apply_leak(self.state, next_state, self.leaking_rate)
        return self.state

    def __lshift__(self, other):
        return self.__call__(other)

    def __rshift__(self, other):
        return default_output(
            self.weights.W_out,
            self.state,
            other,
            self.squeeze_o
            # my_sigm
            # sigmoid
            # self.squeeze_f
        )


def_updator = lambda fun: lambda *args, **kwargs: Updator(fun, *args, **kwargs)


@def_updator
@u.pre_proc_args({
    "state":
    u.force_2dim,
    "tuple":
    lambda tuple: (u.force_2dim(tuple[0]), u.force_2dim(tuple[1]))
})
def vanilla_updator(weights: m.Esn_matrixs, state, tuple):
    (input, _) = tuple
    return weights.W_in.dot(input) + weights.W_res.dot(state)


@def_updator
@u.pre_proc_args({
    "state":
    u.force_2dim,
    "tuple":
    lambda tuple: (u.force_2dim(tuple[0]), u.force_2dim(tuple[1]))
})
def feedback_updator(weights: m.Esn_matrixs, state, tuple):
    (input, output) = tuple
    return ((weights.W_in * input) + (weights.W_res * state) +
            (weights.W_feb * output))


@u.pre_proc_args({"input": u.force_2dim, "state": u.force_2dim})
def default_output(W_out, state, input, output_f=lambda x: x):
    z_n = m.build_extended_states(input.T, state.T).T
    return output_f(W_out.dot(z_n)).reshape(-1)
