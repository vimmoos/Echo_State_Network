import numpy as np
import typing as ty
from dataclasses import dataclass
from functools import reduce
from collections.abc import Generator

comp = lambda *fs: reduce(
    lambda f, g: lambda *args, **kwargs: f(g(*args, **kwargs)), fs)

callable_attr = lambda cls, string: (hasattr(cls, string) and callable(
    getattr(cls, string)))

Comp = lambda fun: lambda inner_fun: comp(fun, inner_fun)


def factories(*args):
    def decorator(_class):
        for method in args:
            setattr(_class, method, comp(_class, getattr(_class, method)))
        return _class

    return decorator


def validate(self, _class):
    for fname, fdef in _class.__dataclass_fields__.items():
        fval = getattr(self, fname)
        if type(fdef.type) == type(lambda x: x):
            if fdef.type(fval):
                continue

            raise ValueError(
                f"The field {fname} failed to be validated with the function {fdef.type}"
            )

        elif isinstance(fval, fdef.type):
            continue

        raise ValueError(f"The field {fname} should be of type {fdef.type}")
    return True


def mydataclass(check=False, **kwargs):
    def decorator(_class):
        if check:
            old_post_init = _class.__post_init__ if hasattr(
                _class, "__post_init__") else lambda self: None
            _class.__post_init__ = lambda self: validate(
                self, _class) and old_post_init(self)

        return dataclass(_class, **kwargs)

    return decorator


def register_methods(kwargs):
    def decorator(_class):
        for k, v in kwargs.items():
            setattr(_class, k, v)
        return _class

    return decorator


@mydataclass(init=True, repr=True)
class Function_wrapper():
    __func: callable

    def __call__(self, *args, **kwargs):
        return self.__func(*args, **kwargs)

    def __and__(self, other):
        if type(other) != type(lambda x: x):
            raise ValueError("the other is not a function cannot comp")
        return comp(self, other)


@mydataclass(repr=True, init=True, check=False)
class My_generator():
    generator: Generator  # solve this !!

    def __enter__(self):
        return self.generator

    def __exit__(self, err_t, err_v, traceback):
        if not isinstance(err_v, StopIteration):
            raise ValueError(f"undefined error ${err_t} ${err_v} ${traceback}")


''' TODO this shouldnt be here and also is still with the old matrix
implementation!! rewrite with the new Matrix interface

'''


def build_extended_states(m_inputs, states, init_len=0, **kwargs):
    return np.vstack((m_inputs.T[:, init_len:], states.T[:, init_len:])).T
