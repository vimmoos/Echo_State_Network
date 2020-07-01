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
<<<<<<< HEAD:project/test/esn_t.py
import project.test.music_test as tgen
import project.parse_midi.matrix.proc_dicts as emidi
=======
import project.music_gen.test as tgen
>>>>>>> b8569a607f5d7aac1cdb9b7ac5d893631de0ea92:project/esn/test.py
import project.parse_midi.matrix.core as cmidi
import project.parse_midi.matrix.proc_dicts as emidi


def multiple__(times=10, thr=lambda x :True):
    def decorator(fun):
        def inner():
            filt = [x for x in [fun()[1] for _ in range(times)] if thr(x)]
            return (times, len(filt), np.mean(filt),filt)

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
        return gen()


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
        return gen()


def test_midi():
    train_len = test_len = 970
    init_len = 100
    music = it.repeat(
        cmidi.exec_proc_dict(emidi.example_proc_dict)["matrixs"][0], 20)
    data = c.Data(np.array(list(music)), None, init_len, train_len, test_len)
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

<<<<<<< HEAD:project/test/esn_t.py
# @multiple__(thr=lambda x : sum(x) > 4)
def test_generated():
    train_len = test_len = 1200
    init_len = 200
=======

def test_generated():
    train_len = test_len = 800
    init_len = 100
>>>>>>> b8569a607f5d7aac1cdb9b7ac5d893631de0ea92:project/esn/test.py

    music = (tgen.test_patterns[2] * 300)

    data = c.Data(np.array(list(~music)), music.tempo, init_len, train_len,
                  test_len)
    with c.Run(
            **{
                "data": data,
                "in_out": 9,
                "leaking_rate": 0.3,
                "reg": 1e-8,
<<<<<<< HEAD:project/test/esn_t.py
                "transformer": ta.Transformers.pow_prob,
                "t_param": 1,
                "t_squeeze": np.tanh,
            }).load("/home/vimmoos/NN/resources/reservoir/0.18333333333333335_0.04_2000",0) as gen:
=======
                "transformer": ta.user_threshold(0.75)
            }) as gen:
>>>>>>> b8569a607f5d7aac1cdb9b7ac5d893631de0ea92:project/esn/test.py
        return gen()


def run_multiple(times, func):
    res_list = [func()[1] for _ in range(times)]
    return np.mean(res_list), res_list


# tot, ind = run_multiple(10, test_generated)
