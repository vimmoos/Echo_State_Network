import itertools as it
from math import inf
from pprint import pprint

import numpy as np
from matplotlib.pyplot import *
from scipy import linalg, sparse

import project.esn.core as c
import project.esn.matrix as m
import project.esn.teacher as te
import project.esn.trainer as tr
import project.esn.transformer as ta
import project.esn.updater as up
import project.music_gen.core as cgen
import project.parse_midi.matrix.core as cmidi
import project.parse_midi.matrix.proc_dicts as emidi
import project.test.music_test as tgen


def multiple__(times=10, thr=lambda x: True):
    def decorator(fun):
        def inner():
            filt = [x for x in [fun()[1] for _ in range(times)] if thr(x)]
            return (times, len(filt), np.mean(filt), filt)

        return inner

    return decorator


def test_generated():
    train_len = test_len = 800
    init_len = 100

    music = (tgen.test_patterns[2] * 300)

    data = c.Data(np.array(list(~music)), music.tempo, init_len, train_len,
                  test_len)
    with c.Run(
            **{
                "data": data,
                "reservoir": 300,
                "in_out": 9,
                "leaking_rate": 0.3,
                "reg": 1e-8,
                "transformer": ta.Transformers.threshold,
                "t_param": 0.75,
                "t_squeeze": ta.my_sigm
            }) as gen:
        return gen()


# tot, ind = run_multiple(10, test_generated)
