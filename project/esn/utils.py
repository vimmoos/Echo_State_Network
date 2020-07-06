import sys
import time
from dataclasses import dataclass
from functools import reduce, wraps
import numpy as np

comp = lambda *fs: reduce(
    lambda f, g: lambda *args, **kwargs: f(g(*args, **kwargs)), fs)

callable_attr = lambda cls, string: (hasattr(cls, string) and callable(
    getattr(cls, string)))

Comp = lambda fun: lambda inner_fun: comp(fun, inner_fun)


def factories(*args):
    """Decorator which call the constructur after the specified methods
are called

    """
    def decorator(_class):
        for method in args:
            setattr(_class, method, comp(_class, getattr(_class, method)))
        return _class

    return decorator


def validate(self, _class):
    """Simple validate function which validate all the value against a
function or a type which is specified as a type annotation

    """
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
    """Enrich the standard dataclass with the check keyword which when
setted to True validate each field value against the corresponding
type annotation

    """
    def decorator(_class):
        if check:
            old_post_init = _class.__post_init__ if hasattr(
                _class, "__post_init__") else lambda self: None
            _class.__post_init__ = lambda self: validate(
                self, _class) and old_post_init(self)

        return dataclass(_class, **kwargs)

    return decorator


def register_methods(kwargs):
    """Register all the methods in the dictionary

    """
    def decorator(_class):
        for k, v in kwargs.items():
            setattr(_class, k, v)
        return _class

    return decorator


def force_2dim(np_arr: np.array):
    """Force 2 dimension on a numpy array

    """
    if np_arr is None: return
    try:
        np_arr.shape[1]
    except IndexError as e:
        return np.reshape(np_arr, (-1, 1))
    return np_arr


force_2dim_all = lambda *args: [force_2dim(x) for x in args]


def pre_proc_args(kwargs):
    """For each key in the dict (where the keys are the parameter that u
want to pre_process) apply the value of that keys and then pass the
output of the application to the specified paramater

    """
    def decorator(fun):
        f_args = fun.__code__.co_varnames
        kw_index = {f_args.index(k): v for k, v in kwargs.items()}

        @wraps(fun)
        def wrapper(*args):
            return fun(*[
                v if i not in kw_index.keys() else kw_index.get(i)(v)
                for i, v in enumerate(args)
            ])

        return wrapper

    return decorator


def signal_hadler():
    """Simple handler which put asleep the process for the specified time
in the std_in

    """
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
