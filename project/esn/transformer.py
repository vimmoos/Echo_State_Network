# import enum as e
from random import uniform

import numpy as np
from aenum import Enum, extend_enum

import project.esn.utils as u

_identity = lambda x: x
_identity.__name__ = "identity"

def sigmoid (x):
    """classical sigmoid function"""
    return 1 / (1 + np.exp(-x))

def enhanced_sigm(x,a):
    """enhanced sigmoid function which scales the input usign the a
    parameter

    """
    return  1 / (1 + np.exp(-(x * a)))

def my_sigm(x):
    """a sigmoid function which is between 0 and 1 and maps 0.5 -> 0.5 ,1
->1 and 0 -> 0

    """
    return enhanced_sigm(x - 0.5, 8)

def squeezed_tanh(x):
    """a squeezed tanh between 0 and 1

    """
    return  (np.tanh(x) + 1) / 2
def choose_prob(x):
    """givend a probability value it returns True if the random number is
below that value otherwise False

    """
    return  uniform(0, 1) <= x


class Squeezers(Enum):
    """collect all the squeezing function

    """
    pass


def fill_squeezer():
    """fill the Squeezers enum using all the possible squeezing function

    """
    for fun in [_identity, sigmoid, my_sigm, squeezed_tanh, np.tanh]:
        extend_enum(Squeezers, fun.__name__, fun)

fill_squeezer()

@np.vectorize
def trim(x):
    """Trim the output between 0 and 1

    """
    x = x if x > 0. else 0.
    return 1. if x > 1. else x


class Transformers(Enum):
    """collect all the possibles Transformer

    """
    pass


@u.mydataclass(init=True, repr=True, frozen=True)
class Transformer():
    """Transformer class : it transform a probability vector to a 0s and
1s

    """
    _transformer: callable = None
    """ the function which perform the transformation
    """
    param: float = 0.0
    """the parameter to the function (it should be used in the decision
process)

    """
    squeeze_f: callable = lambda x: x
    """the squeezing function which squeeze the input of the transformer

    """

    def __call__(self, sig):
        """it trims and squeeze the input signal and then call the
_transformer function

        """
        sig = u.comp(trim, self.squeeze_f)(sig)
        if isinstance(self._transformer, np.vectorize):
            return self._transformer(sig, self.param)
        else:
            return np.array([self._transformer(el, self.param) for el in sig])


def add_transformer(fun):
    """decorator which create a new Transformer and add the new
transformer to the Transformers enum

    """
    inner = lambda *args, **kwargs: Transformer(fun, *args, **kwargs)
    name = fun.pyfunc.__name__ if isinstance(fun,
                                             np.vectorize) else fun.__name__

    extend_enum(Transformers, name, inner)

@add_transformer
@np.vectorize
def identity(x, param):
    """identity function

    """
    return x


@add_transformer
@np.vectorize
def threshold(x, t):
    """use the parameter as a threshold if the value if higher then the
threshold the it return a 1 otherwise 0

    """
    return 1 if x > t else 0


@add_transformer
@np.vectorize
def pow_prob(x, alpha):
    """scale the input non-linearly by alpha and then convert the
probability non-deterministically to 1 or 0

    """
    return 1 if choose_prob(x**alpha) else 0


@add_transformer
@np.vectorize
def sig_prob(x, alpha):
    """substract 0.5 from the input by 0.5 and then scale it by the alpha
parameter. non determistic

    """
    return 1 if choose_prob(sigmoid((x - 0.5) * ((alpha * 10) + 4))) else 0
