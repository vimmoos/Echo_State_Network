import numpy as np
from scipy import sparse, stats, linalg
from project.esn import updater as up
from project.esn import utils as ut
from project.esn import matrix as m


def runner(updator: up.Updator, inputs, outputs=None):
    inputs_len = len(inputs)
    if outputs == None:
        outputs = np.zeros(inputs_len)
    with ut.My_generator(zip(inputs, outputs)) as gen:
        (u, o) = next(gen)
        while True:
            (new_u, new_out) = (yield updator << (u, o))
            (u, o) = next(gen) if new_u == None else (new_u, new_out)


'''TODO maybe its better to collect all in a list and then set the
states

'''


def run_extended(updator: up.Updator, res_size, inputs, outputs=None):
    states = m.zeros((len(inputs) + 1, res_size))
    states[0, :] = updator.state
    for (i, s) in zip(range(len(inputs)), runner(updator, inputs, outputs)):
        states[i + 1, :] = s

    states = states[1:, :]
    return ut.build_extended_states(np.matrix(inputs), states)


# def run_gen_mode(W_in,
#                  W_res,
#                  W_out,
#                  init_input,
#                  init_state,
#                  run_length,
#                  updater=up.leaking_update,
#                  output=up.default_output,
#                  transformer=lambda x: x,
#                  **kwargs):
#     state = init_state
#     input = init_input

#     outputs = np.zeros((run_length, W_out.shape[0]))
#     for t in range(run_length - 1):
#         state = updater(W_in, W_res, input, state, **kwargs)
#         outputs[t, :] = transformer(output(W_out, input, state, **kwargs))
#         input = outputs[t, :]
#     return outputs
''' TODO finish gen_mode first rewrite default_output though!!!
'''


def run_gen_mode(weights: m.Esn_matrixs, updator: up.Updator, init_input,
                          run_length):
    outputs = np.zeros((run_length, weights.W_out.shape()[0]))
    gen_state = runner(updator, np.array([init_input]))
    for (s, t) in zip(gen_state, range(run_length - 1)):
        outputs[t, :] = transformer(up.default_output(weights.W_out, ))
