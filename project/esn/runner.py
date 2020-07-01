import numpy as np
from project.esn.matrix import build_extended_states
from project.esn import updater as up
from project.esn.utils import mydataclass
from project.esn.transformer  import Transformer

from pprint import pprint as p


@mydataclass(init=True, repr=True,frozen=True)
class Runner:
    _runner: callable
    updator: up.Updator
    run_length: int
    inputs: np.ndarray = np.zeros((0, 0))
    outputs: np.ndarray = np.zeros((0, 0))

    def __call__(self,in_out=None):
        inps, outs  = (self.inputs,self.outputs) if in_out is None else in_out
        return self._runner(self.updator,inps,outs)



d_runner = lambda fun: lambda updator,run_length,inputs=None, outputs=None: Runner(
    fun, updator,run_length, inputs, outputs)


@d_runner
def runner(updator: up.Updator, inputs, outputs=None):
    outputs = np.zeros(len(inputs)) if outputs is None else outputs
    gen = zip(inputs, outputs)
    (u, o) = next(gen)
    try:
        while True:
            new_uo = yield updator << (u, o)
            (u, o) = next(gen) if new_uo == None else new_uo
    except StopIteration as e:
        return


def run_extended(r: Runner, init_len=0,in_out=None):
    inputs = r.inputs if in_out is None else in_out[0]
    states = np.zeros((len(inputs) + 1, r.updator.weights.W_res.shape[0]))
    states[0, :] = r.updator.state[:, 0]
    runner = r() if in_out is None else r(in_out)
    for i, s in enumerate(runner):
        states[i + 1, :] = s[:, 0]

    states = states[1:, :]
    return build_extended_states(inputs, states, init_len)


def run_gen_mode(r: Runner, ta: Transformer, input):
    outputs = np.zeros((r.run_length, r.updator.weights.W_out.shape[0]))
    gen_state = r((np.array([input]), None))
    for i in range(r.run_length):
        state = gen_state.send((input, None) if i != 0 else None)
        outputs[i, :] = r.updator >> input
        input = ta(outputs[i,:])

    return outputs
