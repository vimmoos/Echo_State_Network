from pprint import pprint as p

import numpy as np
import project.esn.matrix as m
from project.esn import utils as u


def apply_leak(state, update, leaking_rate=0.3):
    """Simple leaking formula which returns the new state from the next
state and the previous one scaling both by the leaking rate and
1-leaking rate

    """
    return (update * leaking_rate) + (state * (1 - leaking_rate))


@u.mydataclass(repr=True, init=True, check=False)
class Updator():
    """Updator data class This class manage the update process and all
    the weights involved in it It also store the last state in other
    to use it in the next update
    """


    _next: callable = None
    """ The update function used to calculate the next state """

    weights: m.Esn_matrixs = None
    """The class which wraps all the matrixs involved in the ESN"""

    state: np.ndarray = None
    """The current state of the network (it will be updated at each
iteration)"""

    squeeze_f: callable = np.tanh
    """The squeezing function applied to the result of the the _next
function"""

    squeeze_o: callable = lambda x: x
    """The squeezing function applied to the result of the output
function"""

    leaking_rate: float = 0.3
    """The leaking rate used to calculate the _next state"""

    noise: float = 0.0
    """The noise which is added to the result of the _next function"""

    def __post_init__(self):
        """Enrich the specified _next function. It adds the noise and squeeze
the final output after calling the _next function"""
        self._next = u.comp(self.squeeze_f, lambda x: x + self.noise,
                            self._next)

    def __call__(self, other):
        """calculate the next state from a tuple(other) which is composed by
the input and possibly an output

        """
        next_state = self._next(self.weights, self.state, other)
        self.state = apply_leak(self.state, next_state, self.leaking_rate)
        return self.state

    def __lshift__(self, other):
        """reflect the __call__ function

        """
        return self.__call__(other)

    def __rshift__(self, other):
        """calculate the output of the network given an input (other)"""
        return default_output(self.weights.W_out, self.state, other,
                              self.squeeze_o)


def def_updator(fun):
    """Decorator which define a new updater
    """
    def inner(*args, **kwargs):
        return Updator(fun, *args, **kwargs)
    return inner



@def_updator
@u.pre_proc_args({
    "state":
    u.force_2dim,
    "tuple":
    lambda tuple: (u.force_2dim(tuple[0]), u.force_2dim(tuple[1]))
})
def vanilla_updator(weights: m.Esn_matrixs, state, tuple):
    """This calculate the next state using the standard update formula
    \(1+1 \)

    """
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
    """This function calculate the next state given an input and an output
using the enhanced formula

    """
    (input, output) = tuple
    return ((weights.W_in * input) + (weights.W_res * state) +
            (weights.W_feb * output))


@u.pre_proc_args({"input": u.force_2dim, "state": u.force_2dim})
def default_output(W_out, state, input, output_f=lambda x: x):
    """calculate the output using the standard formula

    """
    z_n = m.build_extended_states(input.T, state.T).T
    return output_f(W_out.dot(z_n)).reshape(-1)
