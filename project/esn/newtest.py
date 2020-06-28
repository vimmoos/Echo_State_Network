import numpy as np
from matplotlib.pyplot import *
from scipy import linalg, sparse
import project.esn.matrix as m
import project.esn.updater as up
import project.esn.trainer as tr
import project.esn.core as c
import project.music_gen.core as cgen
import project.esn.transformer as ta
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


def test_generated():
    train_len = test_len = 2400
    data = np.array(list(cgen.note_sampler(cgen.test * 200)))
    with c.Run(
            **{
                "data": data,
                "in_out": 9,
                "reservoir": 300,
                "train_len": train_len,
                "test_len": test_len,
                "init_len": 100,
                "leaking_rate": 0.3,
                "spectral_radius": 0.8,
                "density": .5,
                "reg": 1e-8,
                "transformer": ta.user_threshold(0.5),
            }) as gen:
        return gen()
