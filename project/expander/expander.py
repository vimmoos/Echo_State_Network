import copy
import pickle
import signal

import project.esn.core as c
import project.esn.matrix as m
import project.esn.transformer as t
from project.esn.utils import *
from itertools import product,tee




@mydataclass(init=True, repr=True, check=True)
class Expander():
    """Class that performs a cartesian product of the gen_dict and then
call the _generator function for each configuration the _name_gen if
called witha configuration should return a valid name/path used to
dump the file

    """
    _generator: callable
    _name_gen: callable
    gen_dict: lambda x: True

    def __post_init__(self):
        self._gen_cart = ({
            k: v
            for k, v in zip(self.gen_dict.keys(), elem)
        } for elem in product(*[v for k, v in self.gen_dict.items()]))

    def __call__(self):
        self._gen_cart, copy = tee(self._gen_cart)
        return ({
            **elem,
            "result": self._generator(**elem),
        } for elem in copy)

    def __invert__(self):
        self._gen_cart, copy = tee(self._gen_cart)
        return (self._name_gen(elem) for elem in copy)


def d_expander(inverter):
    """decorator that create a new Expander

    """
    inner = lambda fun: lambda gen_dict, *args, **kwargs: Expander(
        fun, inverter, gen_dict, *args, **kwargs)
    return inner


def res_name(conf: dict):
    """ name generator for the reservoir runs
"""
    return [
        str(v) for k, v in conf.items() if k not in ["result", "repetition"]
    ]


def esn_name(conf: dict):
    """name generator for the esn runs

    """
    return str(
        reduce(lambda x, y: hash(str(hash(str(y))) + str(hash(str(x)))),
               conf.values()))


@d_expander(lambda x: "_".join(res_name(x)))
def gen_reservoir(spectral_radius=None,
                  density=None,
                  size=None,
                  repetition=None):
    """perform the reservoir generation

    """
    return [
        m.scale_spectral_smatrix(m.generate_smatrix(size,
                                                    size,
                                                    density=density),
                                 spectral_radius=spectral_radius)
        for _ in range(repetition)
    ]


@d_expander(esn_name)
def run_esn(repetition, matrix_path, idx, **kwargs):
    """run the esn with the current configuration

    """
    return [
        c.Run(**kwargs).load(matrix_path, idx).__enter__()()
        for _ in range(repetition)
    ] if kwargs["transformer"] not in [
        t.Transformers.threshold, t.Transformers.identity
    ] else [c.Run(**kwargs).load(matrix_path, idx).__enter__()()]


@mydataclass(init=True, repr=True, check=False)
class Pickler():
    """This class wraps an Expander, when called pickle all the results
of the expander process

    """
    expander: Expander
    path_to_dir: str
    _dumper: callable = lambda x: x
    max_exp: lambda x: x is True or isinstance(x, int) = True
    verbose: bool = None

    def __call__(self, max_exp=None):
        if not max_exp is None:
            self.max_exp = max_exp
        for i, conf in enumerate(self.expander()):
            if (not (self.max_exp is True)) and self.max_exp == i:
                return self.path_to_dir
            print(f"dump conf {i}{self._dumper(conf)}" if self.
                  verbose else f"dump conf {i}")
            with open(
                    self.path_to_dir + self.expander._name_gen(conf),
                    "wb") as f:
                pickle.dump(self._dumper(conf), f)

        return self.path_to_dir


reservoir_pickler = lambda gen_dict, *args, **kwargs: Pickler(
    gen_reservoir(gen_dict), *args, **kwargs)

esn_pickler = lambda gen_dict, *args, **kwargs: Pickler(
    run_esn(gen_dict), *args, **kwargs, _dumper=lambda x: x["result"])
