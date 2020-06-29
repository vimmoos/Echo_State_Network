import numpy as np
from matplotlib.pyplot import *
from scipy import linalg, sparse
import project.esn.matrix as m
import project.esn.updater as up
import project.esn.trainer as tr
import project.esn.core as c
import project.esn.transformer as ta
import project.music_gen.core as cgen
import project.music_gen.test as tgen
import project.parse_midi.matrix.proc_dicts as emidi
import project.parse_midi.matrix.core as cmidi
import project.esn.teacher as te
import itertools as it

from math import inf


def multiple__(times=10, thr=inf):
    def decorator(fun):
        def inner():
            filt = [x for x in [fun() for _ in range(times)] if x < thr]
            return (times, len(filt), np.mean(filt))

        return inner

    return decorator


@multiple__(thr=10)
def test():
    data = np.loadtxt('/home/vimmoos/NN/testESN/MackeyGlass_t17.txt')
    with c.Run(
            **{
                "data": data,
                "in_out": 1,
                "reservoir": 100,
                "train_len": 2000,
                "test_len": 2000,
                "init_len": 100,
                "leaking_rate": 0.3,
                "spectral_radius": 1.25,
                "density": .5,
                "reg": 1e-8
            }) as gen:
        Y, mse = gen()
        return mse


@multiple__()
def test_randomMatrix():
    import random
    train_len = test_len = 2000
    data = np.array([
        np.array([random.randint(0, 10) / 10])
        for _ in range(train_len + test_len + 1)
    ])
    with c.Run(
            **{
                "data": data,
                "in_out": 1,
                "reservoir": 100,
                "train_len": train_len,
                "test_len": test_len,
                "init_len": 100,
                "leaking_rate": 0.3,
                "spectral_radius": 1.25,
                "density": .5,
                "reg": 1e-8
            }) as gen:
        Y, mse = gen()
        return mse

def test_midi():
    train_len = test_len = 970
    init_len = 100
    music = it.repeat(cmidi.exec_proc_dict(emidi.example_proc_dict)["matrixs"][0],20)
    data = c.Data(np.array(list(music)),
                  None,
                  init_len,
                  train_len,
                  test_len)
    with c.Run(
            **{
                "data": data,
                "in_out": 6,
                "reservoir": 500,
                "error_len": 500,
                "leaking_rate": 0.3,
                "spectral_radius": 0.8,
                "density": .5,
                "reg": 1e-8,
                "transformer": ta.user_threshold(0.5),
            }) as gen:
        return gen()

def test_generated():
    train_len = test_len = 1200
    init_len = 100

    music = (tgen.test * 200)

    data = c.Data(np.array(list(~music)),
                music.tempo,
                init_len,
                train_len,
                test_len)
    with c.Run(
            **{
                "data": data,
                "in_out": 9,
                "reservoir": 500,
                "error_len": 500,
                "leaking_rate": 0.3,
                "spectral_radius": 0.8,
                "density": .5,
                "reg": 1e-8,
                "transformer": ta.user_threshold(0.3)
            }) as gen:
        return gen()
