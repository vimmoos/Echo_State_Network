from pprint import pprint as p

import numpy as np
from scipy import linalg, sparse, stats

import project.esn.transformer as tr
from project.esn import matrix as m
from project.esn import runner as r
from project.esn import teacher as te
from project.esn import trainer as t
from project.esn import updater as up
from project.esn import utils as ut
from project.music_gen.data_types import Tempo


@ut.mydataclass(init=True, repr=True, check=False)
class ESN:
    _runner: r.Runner
    trainer: t.Trainer
    transformer: callable
    init_len: int = 100

    def __lshift__(self, other):
        (input, desired) = other
        ex_state = r.run_extended(self._runner << (input, None), self.init_len)
        self._runner.updator.weights.W_out = self.trainer << (ex_state,
                                                              desired)
        return ex_state

    def __rshift__(self, other):
        return r.run_gen_mode(self._runner, self.transformer, other)


@ut.mydataclass(init=True, repr=True, check=False)
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


@ut.mydataclass(init=True, repr=True, check=False)
class Run:
    data: Data
    in_out: int = 1
    reservoir: int = 100
    error_len: int = 500
    leaking_rate: float = 0.3
    spectral_radius: float = 0.9
    density: float = 0.5
    reg: float = 1e-8
    transformer: callable = lambda x: x
    evaluation: callable = te._mse_nd

    def __call__(self, input=None, desired=None):
        input = self.data.start_input() if input is None else input
        desired = self.data.test_data() if desired is None else desired
        output = self.esn >> input
        return (output,
                self.evaluation(output, ut.force_2dim(desired),
                                self.error_len), self.data.tempo)

    def __enter__(self):
        self.init_len = self.data.init_len
        matrixs = m.esn_matrixs(
            m.generate_rmatrix(
                self.reservoir,
                self.in_out,
                spectral_radius=self.spectral_radius,
                density=self.density,
            ))

        updator = up.vanilla_updator(matrixs,
                                     np.zeros((self.reservoir, 1)),
                                     leaking_rate=self.leaking_rate)

        trainer = t.ridge_reg(param=self.reg)

        self.esn = ESN(r.runner(updator, self.data.test_len), trainer,
                       self.transformer, self.init_len)
        self.activations = self.esn << (self.data.training_data(),
                                        self.data.desired())
        return self

    def __exit__(self, err_t, err_v, traceback):
        pass
