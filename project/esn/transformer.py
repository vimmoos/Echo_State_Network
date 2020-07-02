from random import uniform

import numpy as np

import project.esn.utils as u
from aenum import Enum, extend_enum

_identity = lambda x : x
_identity.__name__ = "identity"


sigmoid = lambda x: 1 / (1 + np.exp(-x))
sigmoid.__name__ = "sigmoid"

enhanced_sigm = lambda x, a: 1 / (1 + np.exp(-(x * a)))
enhanced_sigm.__name__ = "enhanced_sigm"

my_sigm = lambda x: enhanced_sigm(x - 0.5, 8)
my_sigm.__name__ = "my_sigm"

squeezed_tanh = lambda x: (np.tanh(x) + 1) / 2
squeezed_tanh.__name__ = "squeezed_tanh"

choose_prob = lambda x: uniform(0, 1) <= x


@np.vectorize
def trim(x):
    x = x if x > 0. else 0.
    return 1. if x > 1. else x


class Transformers(Enum):
    pass


@u.mydataclass(init=True, repr=True, frozen=True)
class Transformer():
    _transformer: callable
    param: float = 0.0
    squeeze_f: callable = lambda x: x

    def __call__(self, sig):
        sig = u.comp(trim, self.squeeze_f)(sig)
        if isinstance(self._transformer, np.vectorize):
            return self._transformer(sig, self.param)
        else:
            return np.array([self._transformer(el, self.param) for el in sig])


def add_transformer(fun):
    inner = lambda *args, **kwargs: Transformer(fun, *args, **kwargs)
    name = fun.pyfunc.__name__ if isinstance(fun,
                                             np.vectorize) else fun.__name__

    extend_enum(Transformers, name, inner)


# @add_transformer
# def softmax(li, alpha):
#     return np.array([
#         1 if choose_prob(x**alpha) else 0
#         for x in (np.exp(li) / sum(np.exp(li)))
#     ])


@add_transformer
@np.vectorize
def identity(x, param):
    return x


@add_transformer
@np.vectorize
def threshold(x, t):
    return 1 if x > t else 0


@add_transformer
@np.vectorize
def pow_prob(x, alpha):
    return 1 if choose_prob(x**alpha) else 0


@add_transformer
@np.vectorize
def sig_prob(x, alpha):
    return 1 if choose_prob(sigmoid((x - 0.5) * ((alpha * 10) + 4))) else 0
