import numpy as np
from scipy import sparse, stats, linalg
from project.esn import updater as up
from project.esn import utils as ut
from project.esn import matrix as m
from project.esn import trainer as t
import project.esn.transformer as tr


@ut.mydataclass(init=True, repr=True)
class Runner():
    _runner: callable
    updator: up.Updator
    run_length: int
    inputs: np.ndarray = np.zeros((0, 0))
    outputs: np.ndarray = np.zeros((0, 0))

    def __call__(self):
        return self._runner(self.updator, self.inputs, self.outputs)

    def __lshift__(self, other):
        (inps, outs) = other
        self.inputs = inps
        self.outputs = outs
        return self


d_runner = lambda fun: lambda updator, inputs, outputs=None: Runner(
    fun, updator, inputs, outputs)


@d_runner
def runner(updator: up.Updator, inputs, outputs=None):
    outputs = np.zeros(len(inputs)) if outputs is None else outputs
    gen = zip(inputs, outputs)
    (u, o) = next(gen)
    try:
        while True:
            new_uo = (yield updator << (u, o))
            (u, o) = next(gen) if new_uo == None else new_uo
    except StopIteration as e:
        return


def run_extended(r: Runner, init_len=0):
    states = np.zeros((len(r.inputs) + 1, r.updator.weights.W_res.shape[0]))
    states[0, :] = r.updator.state[:, 0]
    for (i, s) in zip(range(len(r.inputs)), r()):
        states[i + 1, :] = s[:, 0]

    states = states[1:, :]
    return m.build_extended_states(r.inputs, states, init_len)


def run_gen_mode(r: Runner, ta: callable, input):
    outputs = np.zeros((r.run_length, r.updator.weights.W_out.shape[0]))
    gen_state = (r << (np.array([input]), None))()
    for i in range(r.run_length):
        state = gen_state.send((input, None) if i != 0 else None)
        input = ta(r.updator >> input)
        outputs[i, :] = input

    return outputs


from pprint import pprint as p


@ut.mydataclass(init=True, repr=True, check=False)
class ESN():
    _runner: Runner
    trainer: t.Trainer
    transformer: callable
    init_len: int = 100

    def __lshift__(self, other):
        (input, desired) = other
        ex_state = run_extended(self._runner << (input, None), self.init_len)
        self._runner.updator.weights.W_out = self.trainer << (ex_state,
                                                              desired)
        return ex_state

    def __rshift__(self, other):
        return run_gen_mode(self._runner, self.transformer, other)


@ut.mydataclass(init=True, repr=True, check=False)
class Run():
    data: np.ndarray
    in_out: int = 1
    reservoir: int = 100
    train_len: int = 2000
    test_len: int = 2000
    init_len: int = 100
    error_len: int = 500
    leaking_rate: float = 0.3
    spectral_radius: float = 0.9
    density: float = 0.5
    reg: float = 1e-8
    transformer: callable = lambda x: x

    def __call__(self, input=None, desired=None):
        input = self.data[self.train_len] if input is None else input
        desired = self.data[self.train_len +
                            1:] if desired is None else desired
        output = self.esn >> input
        return (output, self._mse_nd(output, ut.force_2dim(desired)))

    def _mse1d(self, output, desired):
        return sum(
            np.square(desired[:self.error_len] -
                      output[:self.error_len])) / self.error_len

    def _mse_nd(self, output, desired):
        return [
            self._mse1d(output[:,x],desired[:,x])
            for x in range(output.shape[1])
        ]

    def __enter__(self):
        matrixs = m.esn_matrixs(
            m.generate_rmatrix(self.reservoir,
                               self.in_out,
                               spectral_radius=self.spectral_radius,
                               density=self.density))
        updator = up.vanilla_updator(matrixs,
                                     np.zeros((self.reservoir, 1)),
                                     leaking_rate=self.leaking_rate)
        trainer = t.ridge_reg(param=self.reg)
        desired = self.data[self.init_len + 1:self.train_len + 1]
        self.esn = ESN(runner(updator, self.test_len), trainer,
                       self.transformer)
        self.activations = self.esn << (self.data[:self.train_len], desired)
        return self

    def __exit__(self, err_t, err_v, traceback):
        pass
