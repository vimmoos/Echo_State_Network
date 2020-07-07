import enum as e

import numpy as np

import project.esn.core as core
import project.esn.transformer as transf
import project.test.music_test as tmusic
import project.PSO.Landscape as l
import signal as s
import project.esn.utils as u

# test_dimensions = {"leaking_rate": (0.05, 1), "spectral_radius": (0.05, 1.5)}
# test_dimensions = {"transformer": transf.Transformers}
path = "/home/vimmoos/NN/resources/reservoir/final/"
path_esn = "/home/vimmoos/NN/resources/esn/final_pso/"


class Res(e.Enum):
    res_0 = "1.1_0.04_2000"
    res_1 = "0.9_0.04_5000"
    res_2 = "0.5_0.04_2000"
    res_3 = "0.5_0.04_5000"
    res_4 = "0.7_0.04_2000"
    res_5 = "0.7_0.04_5000"
    res_6 = "0.9_0.04_2000"
    res_7 = "1.1_0.04_5000"
    res_8 = "1.3_0.04_2000"
    res_9 = "1.3_0.04_5000"


test_dimensions = {
    "transformer": transf.Transformers,
    "squeeze_o": transf.Squeezers,
    "t_squeeze": transf.Squeezers,
    "t_param": (0.0005, 1),
    "leaking_rate": (0.0005, 1),
    "noise": (0.0005, 1),
    "reservoir": Res,
}

esn_gen = {
    "data":
    core.Data(np.array(list(~(tmusic.new_all * 10))), (tmusic.new_all * 10).tempo,
              3200, 6600, 6600),
    "density":
    0.04,
    "in_out":
    9,
    "reg":
    1e-8,
}

s.signal(s.SIGINT, u.signal_hadler())
