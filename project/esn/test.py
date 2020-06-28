import numpy as np
from project.esn import matrix as m
from project.esn import updater as up
from project.esn import trainer as t
from project.esn import core as c
from project.esn import utils as ut
from project.esn import transformer as tr
from project.parse_midi.matrix import core as parse
from project.parse_midi.matrix import proc_dicts as midi
from scipy import linalg
from pprint import pprint as p

# def test_MackeyGlass():
#     trainLen = 2000
#     testLen = 2000
#     initLen = 100
#     data = np.loadtxt('/home/vimmoos/NN/testESN/MackeyGlass_t17.txt')
#     print(data.shape)

#     inSize = outSize = 1
#     resSize = 100
#     Yt = data[initLen + 1:trainLen + 1, None]
#     np.random.seed(42)
#     Win = np.random.rand(resSize, inSize) - 0.5

#     W = m.generate_smatrix(resSize, resSize)
#     m.scale_spectral_smatrix(W, in_place=True, spectral_radius=1.25)

#     print("run states")
#     x1 = c.run_states(Win, W, data[:trainLen, None],
#                       np.array([0 for _ in range(resSize)]))

#     inputs = np.matrix(np.array([[x] for x in data[:trainLen]]))
#     print("concatenate input and state")
#     X1 = ut.build_extended_states(inputs, x1, initLen)

#     print("calculate Wout")
#     Wout = t.ridge_reg(X1, Yt, 1e-8)

#     print("run gen mode")
#     Y1 = c.run_gen_mode(Win, W, Wout, np.array([data[trainLen]]), x1[-1, :],
#                         testLen)

#     print("calculate mse")
#     errorLen = 500
#     mse1 = sum(
#         np.square(data[trainLen + 1:trainLen + errorLen + 1] -
#                   Y1.T[0, 0:errorLen])) / errorLen
#     print('MSE1 = ' + str(mse1))

# # load the data
# trainLen = 2000
# testLen = 2000
# initLen = 100
# data = np.loadtxt('MackeyGlass_t17.txt')

# # plot some of it
# figure(10).clear()
# plot(data[0:1000])
# title('A sample of data')


def new_t():
    trainLen = 2000
    testLen = 2000
    initLen = 100
    data = np.loadtxt('/home/vimmoos/NN/testESN/MackeyGlass_t17.txt')
    print(data.shape)
    # generate the ESN reservoir
    inSize = outSize = 1
    resSize = 100
    a = 0.3  # leaking rate
    matrixs = m.esn_matrixs(m.generate_rmatrix(resSize, inSize))
    Win = (np.random.rand(resSize, 1 + inSize) - 0.5) * 1
    W = np.random.rand(resSize, resSize) - 0.5
    # normalizing and setting spectral radius (correct, slow):
    print('Computing spectral radius...'),
    rhoW = max(abs(linalg.eig(W)[0]))
    print('done.')
    W *= 1.25 / rhoW

    # allocated memory for the design (collected states) matrix
    X0 = np.zeros((inSize + resSize, trainLen))
    # set the corresponding target matrix directly
    Yt = data[None, 1:trainLen + 1]
    # updator = up.vanilla_updator(matrixs,
    #                              np.zeros((resSize, )),
    #                              squeeze_f=np.tanh,
    #                              leaking_rate=a)
    # runner = c.runner(updator, testLen)
    # run the reservoir with the data and collect X
    # X = c.run_extended(runner << (data[:trainLen, None], None))
    x = np.zeros((resSize, 1))
    for t in range(trainLen):
        u = data[t]
        x = (1 - a) * x + a * np.tanh(np.dot(Win, u) + np.dot(W, x))
        X0[:, t] = np.vstack((u, x))[:, 0]

    # p(X.T)
    # p(X0)
    # p(X.T.shape)
    # p(X0.shape)

    # # train the output by ridge regression
    # X_T = X.T
    # Wout = np.dot(np.dot(Yt,X), linalg.inv(np.dot(X.T,X) + \
        #                                        reg*np.eye(inSize+resSize)))
    reg = 1e-8  # regularization coefficient
    X0_T = X0.T
    Wout = np.dot(np.dot(Yt,X0_T), linalg.inv(np.dot(X0,X0_T) + \
                                              reg*np.eye(inSize+resSize)))

    # run the trained ESN in a generative mode. no need to initialize here,
    # because x is initialized with training data and we continue from there.
    Y = np.zeros((outSize, testLen))
    u = data[trainLen]
    # x = updator.state[:, None]
    for t in range(testLen):
        x = (1 - a) * x + a * np.tanh(np.dot(Win, u) + W.dot(x))
        y = np.dot(Wout, np.vstack((u, x)))
        Y[:, t] = y
        # generative mode:
        u = y
        ## this would be a predictive mode:
        #u = data[trainLen+t+1]

    # compute MSE for the first errorLen time steps
    errorLen = 500
    mse = sum(
        np.square(data[trainLen + 1:trainLen + errorLen + 1] -
                  Y[0, 0:errorLen])) / errorLen
    print('MSE = ' + str(mse))


def test_MackeyGlass():
    trainLen = 2000
    testLen = 2000
    initLen = 100
    data = np.loadtxt('/home/vimmoos/NN/testESN/MackeyGlass_t17.txt')
    print(data.shape)
    inSize = outSize = 1
    resSize = 100
    matrixs = m.esn_matrixs(m.generate_rmatrix(resSize, inSize))
    updator = u.vanilla_updator(matrixs, np.array([0 for _ in range(resSize)]))
    runner = c.runner(updator, testLen)
    trainer = t.ridge_reg()
    # transformer = tr.user_threshold(0.5)
    transformer = np.vectorize(lambda x: x)
    esn = c.ESN(runner, trainer, transformer)
    des = data[1:trainLen + 1, None]
    esn << (data[:trainLen, None], des)
    outputs = esn >> data[trainLen]
    print(outputs.shape)
    errorLen = 500
    mse1 = sum(
        np.square(data[trainLen + 1:trainLen + errorLen + 1] -
                  outputs.T[0, 0:errorLen])) / errorLen
    print('MSE1 = ' + str(mse1))


# def test_randomMatrix():
#     trainLen = 1000
#     testLen = 1000
#     initLen = 100
#     import random
#     random.seed(42)
#     data = np.array([
#         np.array([random.randint(0, 10)])
#         for _ in range(trainLen + testLen + 1)
#     ])
#     print(data.shape)

#     inSize = outSize = 1
#     resSize = 100
#     Yt = data[initLen + 1:trainLen + 1]
#     np.random.seed(42)
#     Win = np.random.rand(resSize, inSize) - 0.5

#     W = m.generate_smatrix(resSize, resSize)
#     m.scale_spectral_smatrix(W, in_place=True)

#     print("run states")
#     x1 = c.run_states(Win, W, data[:trainLen],
#                       np.array([0 for _ in range(resSize)]))

#     print(x1.shape)

#     inputs = np.matrix(np.array([[x] for x in data[:trainLen]])).T
#     print(inputs.shape)
#     X1 = ut.build_extended_states(inputs, x1, initLen)

#     print(X1.shape)
#     print(Yt.shape)
#     print("calculate Wout")
#     Wout = t.ridge_reg(X1, Yt, 1e-8)

#     print(Wout.shape)
#     print("run gen mode")
#     print(np.array([data[trainLen]]).shape)
#     Y1 = c.run_gen_mode(Win, W, Wout, np.array([data[trainLen, 0]]), x1[-1, :],
#                         testLen)

#     print("calculate mse")
#     errorLen = 500
#     print(data[trainLen + 1:trainLen + errorLen + 1].shape,
#           Y1.T[0, 0:errorLen].shape)
#     mse1 = sum(
#         np.square(data[:, 0][trainLen + 1:trainLen + errorLen + 1] -
#                   Y1.T[0, 0:errorLen])) / errorLen
#     print('MSE1 = ' + str(mse1))
#     return mse1
