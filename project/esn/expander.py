from itertools import product, tee
import pickle
from project.esn.utils import *
import project.esn.matrix as m


@mydataclass(init=True, repr=True, check=True)
class Expander():
    _generator: callable
    gen_dict: lambda x: True

    def __post_init__(self):
        self._gen_cart = ({
            k: v
            for k, v in zip(self.gen_dict.keys(), elem)
        } for elem in (x
                       for x in product(*[v
                                          for k, v in self.gen_dict.items()])))

    def __call__(self):
        self._gen_cart, copy = tee(self._gen_cart)
        return ({
            **elem,
            "result": self._generator(**elem),
        } for elem in copy)


d_expander = lambda fun: lambda gen_dict, *args, **kwargs: Expander(
    fun, gen_dict, *args, **kwargs)


@d_expander
def gen_reservoir(spectral_radius=None,
                  density=None,
                  size=None,
                  repetition=None):
    return [
        m.scale_spectral_smatrix(m.generate_smatrix(size,
                                                    size,
                                                    density=density),
                                 spectral_radius=spectral_radius)
        for _ in range(repetition)
    ]


@mydataclass(init=True, repr=True, check=False)
class Pickler():
    _name_gen: callable
    expander: Expander
    path_to_dir: str
    max_exp: lambda x: x is True or isinstance(x, int) = True

    def __call__(self, max_exp=None):
        if not max_exp is None:
            self.max_exp = max_exp
        for i, conf in enumerate(self.expander()):
            if (not (self.max_exp is True)) and self.max_exp == i:
                return self.path_to_dir
            print(f"dump conf {i}{conf}")
            with open(self.path_to_dir + "_".join(self._name_gen(conf)),
                      "wb") as f:
                pickle.dump(conf, f)

        return self.path_to_dir


reservoir_pickler = lambda fun: lambda gen_dict, *args, **kwargs: Pickler(
    fun, gen_reservoir(gen_dict), *args, **kwargs)


@reservoir_pickler
def vanilla_pickler(conf: dict):
    return [
        str(v) for k, v in conf.items() if not k in ["result", "repetition"]
    ]


reservoir_gen = {
    "spectral_radius": ((x / 20) + 0.05 for x in range(12)),
    "density": ((x / 25) + 0.04 for x in range(12)),
    "size": ((x * 100) * 4 + 400 for x in range(12)),
    "repetition": [10],
}

if __name__ == "__main__":
    p = vanilla_pickler(gen_dict=reservoir_gen,
                        path_to_dir="/home/vimmoos/NN/resources/reservoir/",
                        max_exp=True)
    p()
