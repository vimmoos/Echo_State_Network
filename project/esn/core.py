from pprint import pprint as p

import numpy as np

import project.esn.transformer as tr
from project.esn import matrix as m
from project.esn import runner as r
from project.esn import trainer as t
from project.esn import updater as up
from project.esn import utils as ut
from project.music_gen.data_types import Tempo


@ut.mydataclass(init=True, repr=True, check=False, frozen=True)
class ESN:
    _runner: r.Runner
    trainer: t.Trainer
    transformer: tr.Transformer
    init_len: int = 100

    def __lshift__(self, other):
        (input, desired) = other
        ex_state = r.run_extended(self._runner, self.init_len, (input, None))
        self._runner.updator.weights.W_out = self.trainer((ex_state, desired))
        return ex_state

    def __rshift__(self, other):
        return r.run_gen_mode(self._runner, self.transformer, other)


@ut.mydataclass(init=True, repr=True, check=False, frozen=True)
class Data:
    data: np.ndarray
    tempo: Tempo
    init_len: int
    train_len: int
    test_len: int

    def desired(self):
        return self.data[self.init_len + 1:self.train_len + 1]

    def training_data(self):
        return self.data[:self.train_len]

    def test_data(self):
        return self.data[self.train_len + 1:]

    def start_input(self):
        return self.data[self.train_len]


def _get_val(x):
    if isinstance(x, tr.Transformers):
        return x.name
    return x.__name__ if callable(x) else x


@ut.mydataclass(init=True, repr=True, check=False)
class Run:
    data: Data
    in_out: int = 1
    reservoir: int = 100
    leaking_rate: float = 0.3
    spectral_radius: float = 0.9
    density: float = 0.5
    reg: float = 1e-8
    transformer: tr.Transformers = tr.Transformers.identity
    t_param: float = 0.0
    t_squeeze: callable = np.tanh
    squeeze_o:callable = lambda x :x
    noise: float = 0.0

    def to_dict(self, kwargs={}):
        return {
            **{
                k: _get_val(getattr(self, k))
                for k, _ in self.__class__.__dict__["__dataclass_fields__"].items(
                ) if k not in ["data", "in_out"]
            },
            **kwargs
        }

    def __call__(self, input=None, desired=None):
        input = self.data.start_input() if input is None else input
        desired = self.data.test_data() if desired is None else desired
        output = self.esn >> input
        return self.to_dict({
            "input": input,
            "output": output,
            "desired": desired,
            "tempo": self.data.tempo
        })

    def load(self, path, idx):
        def matrixs_gen(self):
            dic = m.load_smatrix(path, idx)
            self.spectral_radius = dic["spectral_radius"]
            self.density = dic["density"]
            self.reservoir = dic["W_res"].shape[0]
            return m.Esn_matrixs(m.generate_rmatrix(self.reservoir,
                                                    self.in_out),
                                 dic["W_res"],
                                 spectral_radius=self.spectral_radius,
                                 density=self.density)

        setattr(self, "matrixs_gen", lambda: matrixs_gen(self))
        return self

    def matrixs_gen(self):
        return m.esn_matrixs(
            m.generate_rmatrix(self.reservoir, self.in_out),
            spectral_radius=self.spectral_radius,
            density=self.density,
        )

    def __enter__(self):
        self.init_len = self.data.init_len
        matrixs = self.matrixs_gen()

        updator = up.vanilla_updator(matrixs,
                                     np.zeros((self.reservoir, 1)),
                                     leaking_rate=self.leaking_rate,
                                     squeeze_o = self.squeeze_o)

        trainer = t.ridge_reg(param=self.reg)
        transformer = self.transformer.value(self.t_param, self.t_squeeze)

        self.esn = ESN(r.runner(updator, self.data.test_len), trainer,
                       transformer, self.init_len)
        self.activations = self.esn << (self.data.training_data(),
                                        self.data.desired())
        return self

    def __exit__(self, err_t, err_v, traceback):
        pass
