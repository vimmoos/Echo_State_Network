import itertools as it

import numpy as np

import project.esn.core as c
import project.esn.transformer as ta
import project.parse_midi.matrix.core as cmidi
import project.parse_midi.matrix.proc_dicts as emidi
import project.test.music_test as tgen


def test():
    data = np.loadtxt("/home/vimmoos/NN/testESN/MackeyGlass_t17.txt")

    data = c.Data(data, None, 100, 2000, 2000)
    with c.Run(
        **{
            "data": data,
            "in_out": 1,
            "reservoir": 100,
            "leaking_rate": 0.3,
            "spectral_radius": 1.25,
            "density": 0.5,
            "reg": 1e-8,
        }
    ) as gen:
        return gen()


def test_randomMatrix():
    import random

    train_len = test_len = 2000
    data = c.Data(
        np.array(
            [
                np.array([random.randint(0, 10) / 10])
                for _ in range(train_len + test_len + 1)
            ]
        ),
        None,
        100,
        train_len,
        test_len,
    )
    with c.Run(
        **{
            "data": data,
            "in_out": 1,
            "reservoir": 100,
            "leaking_rate": 0.3,
            "spectral_radius": 1.25,
            "density": 0.5,
            "reg": 1e-8,
        }
    ) as gen:
        return gen()


def test_midi():
    train_len = test_len = 970
    init_len = 100
    music = it.repeat(cmidi.exec_proc_dict(emidi.example_proc_dict)["matrixs"][0], 20)
    np_music = np.array(list(music))
    np_music = np_music.reshape((np.prod(np_music.shape[:2]), np_music.shape[-1]))
    data = c.Data(np_music, None, init_len, train_len, test_len)
    with c.Run(
        **{
            "data": data,
            "in_out": 9,
            "reservoir": 500,
            "leaking_rate": 0.3,
            "spectral_radius": 0.8,
            "density": 0.5,
            "reg": 1e-8,
            "transformer": ta.Transformers.threshold,
            "t_param": 0.5,
        }
    ) as gen:
        return gen()


def test_generated():
    train_len = test_len = 1200
    init_len = 200
    music = tgen.test_patterns[2] * 300
    data = c.Data(np.array(list(~music)), music.tempo, init_len, train_len, test_len)
    with c.Run(
        **{
            "data": data,
            "reservoir": 300,
            "in_out": 9,
            "leaking_rate": 0.3,
            "reg": 1e-8,
            "transformer": ta.Transformers.pow_prob,
            "t_param": 1,
            "t_squeeze": np.tanh,
        }
    ) as gen:
        return gen()
