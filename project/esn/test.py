import numpy as np
from project.esn import matrix as m
from project.esn import updater as u
from project.esn import trainer as t
from project.esn import core as c
from project.esn import utils as ut
from project.parse_midi.matrix import core as parse
from project.parse_midi.matrix import proc_dicts as midi
from project.esn import dict_esn as d


def test_MackeyGlass():
    trainLen = 2000
    testLen = 2000
    initLen = 100
    data = np.loadtxt('/home/vimmoos/NN/testESN/MackeyGlass_t17.txt')
    print(data.shape)

    inSize = outSize = 1
    resSize = 100
    Yt = data[initLen + 1:trainLen + 1, None]
    np.random.seed(42)
    Win = np.random.rand(resSize, inSize) - 0.5

    W = m.generate_smatrix(resSize, resSize)
    m.scale_spectral_smatrix(W, in_place=True, spectral_radius=1.25)

    print("run states")
    x1 = c.run_states(Win, W, data[:trainLen, None],
                      np.array([0 for _ in range(resSize)]))

    inputs = np.matrix(np.array([[x] for x in data[:trainLen]]))
    print("concatenate input and state")
    X1 = ut.build_extended_states(inputs, x1, initLen)

    print("calculate Wout")
    Wout = t.ridge_reg(X1, Yt, 1e-8)

    print("run gen mode")
    Y1 = c.run_gen_mode(Win, W, Wout, np.array([data[trainLen]]), x1[-1, :],
                        testLen)

    print("calculate mse")
    errorLen = 500
    mse1 = sum(
        np.square(data[trainLen + 1:trainLen + errorLen + 1] -
                  Y1.T[0, 0:errorLen])) / errorLen
    print('MSE1 = ' + str(mse1))


def test_randomMatrix():
    trainLen = 1000
    testLen = 1000
    initLen = 100
    import random
    random.seed(42)
    data = np.array([
        np.array([random.randint(0, 10)])
        for _ in range(trainLen + testLen + 1)
    ])
    print(data.shape)

    inSize = outSize = 1
    resSize = 100
    Yt = data[initLen + 1:trainLen + 1]
    np.random.seed(42)
    Win = np.random.rand(resSize, inSize) - 0.5

    W = m.generate_smatrix(resSize, resSize)
    m.scale_spectral_smatrix(W, in_place=True)

    print("run states")
    x1 = c.run_states(Win, W, data[:trainLen],
                      np.array([0 for _ in range(resSize)]))

    print(x1.shape)

    inputs = np.matrix(np.array([[x] for x in data[:trainLen]])).T
    print(inputs.shape)
    X1 = ut.build_extended_states(inputs, x1, initLen)

    print(X1.shape)
    print(Yt.shape)
    print("calculate Wout")
    Wout = t.ridge_reg(X1, Yt, 1e-8)

    print(Wout.shape)
    print("run gen mode")
    print(np.array([data[trainLen]]).shape)
    Y1 = c.run_gen_mode(Win, W, Wout, np.array([data[trainLen, 0]]), x1[-1, :],
                        testLen)

    print("calculate mse")
    errorLen = 500
    print(data[trainLen + 1:trainLen + errorLen + 1].shape,
          Y1.T[0, 0:errorLen].shape)
    mse1 = sum(
        np.square(data[:, 0][trainLen + 1:trainLen + errorLen + 1] -
                  Y1.T[0, 0:errorLen])) / errorLen
    print('MSE1 = ' + str(mse1))
    return mse1
