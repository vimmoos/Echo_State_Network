import numpy as np
from dataclasses import dataclass
from functools import reduce
from functools import wraps
import sys
import time

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
        if callable(fdef.type):
            if fdef.type(fval):
                continue

            raise ValueError(
                f"The field {fname} failed to be validated with the function {fdef.type}"
            )

        elif isinstance(fval, type(fdef.type)):
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

def force_2dim(np_arr: np.array):
    if np_arr is None: return
    try:
        np_arr.shape[1]
    except IndexError as e:
        return np.reshape(np_arr, (-1, 1))
    return np_arr


force_2dim_all = lambda *args: [force_2dim(x) for x in args]


def pre_proc_args(kwargs):
    def decorator(fun):
        f_args = fun.__code__.co_varnames
        kw_index = {f_args.index(k): v for k, v in kwargs.items()}

        @wraps(fun)
        def wrapper(*args):
            return fun(*[
                v if not i in kw_index.keys() else kw_index.get(i)(v)
                for i, v in enumerate(args)
            ])

        return wrapper

    return decorator


def signal_hadler():
    count = 0

    def signal_hadler(sig, frame):
        nonlocal count
        count += 1
        if count == 2:
            sys.exit(0)
        print("te volessi mazarme ah")
        print("quanti secondi te me fa dormir?")
        sys.stdout.flush()
        time.sleep(int(input()))
        count = 0

    return signal_hadler


''' TODO '''


def composable(fun):
    setattr(
        fun, "compose", lambda cl, : lambda inner: lambda *args, **kwargs: cl(
            inner, fun(), *args, **kwargs))
