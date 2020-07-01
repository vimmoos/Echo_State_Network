from project.esn.transformer import Transformers, sigmoid, my_sigm
from project.esn.utils import *
import signal
import numpy as np
import project.expander.expander as e

reservoir_gen = {
                 "spectral_radius": ((x / 10) + 0.05 for x in range(12)),
                 "density": ((x / 12) + 0.04 for x in range(12)),
                 "size": (x for x in [100, 500, 1000, 1500, 2000, 3000, 5000, 10000]),
                 "repetition": [10],
                 }

reservoir_gen_small = {
    "spectral_radius": ((x / 12) + 0.1 for x in range(12)),
    "density": ((x / 12) + 0.04 for x in range(10)),
    "size": (x for x in [100, 500, 1000, 2000]),
    "repetition": [10],
}

import project.test.music_test as data
import project.esn.core as c

esn_gen = {
    "data":
    [c.Data(np.array(list(~(data.test * 200))), (data.test * 200).tempo, 216,
           3400, 3400)],
    "in_out": [9],
    "leaking_rate": (x / 10 for x in range(10)),
    "reg": [1e-8],
    "transformer": (x for x in list(Transformers)),
    "t_param": ((x * 2) / 10 for x in range(5)),
    "t_squeeze": (x for x in [np.tanh, sigmoid, my_sigm]),
    "noise": [0],
    "matrix_path":
    ("/".join(["/home", "vimmoos", "NN", "resources", "reservoir", x])
     for x in ~e.gen_reservoir(reservoir_gen_small)),
    "idx": (x for x in range(10)),
    "repetition": [10]
}


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_hadler())
    p = e.esn_pickler(esn_gen,
                      path_to_dir= "/home/vimmoos/NN/resources/esn/")
    p()
