
import numpy as np
from project.esn import matrix as m
from project.esn import updater as u
from project.esn import trainer as t
from project.esn import core as c

def test_MackeyGlass():
    trainLen = 2000
    testLen = 2000
    initLen = 100
    data = np.loadtxt('/home/vimmoos/NN/testESN/MackeyGlass_t17.txt')
    print(data.shape)

    inSize = outSize = 1
    resSize = 100
    Yt = data[None, initLen + 1:trainLen + 1]
    np.random.seed(42)
    Win = np.random.rand(resSize, inSize) - 0.5

    W = m.generate_smatrix(resSize,resSize)
    m.scale_spectral_smatrix(W,in_place =True)


    x1 = c.run_states(Win,W,data[:trainLen,None],
                      np.array([ 0 for _ in range(resSize)]))


    inputs = np.matrix(np.array([[x]for x in data[:trainLen]]))
    X1 = c.build_extended_states(inputs,x1,initLen)

    Wout = t.ridge_reg(X1,Yt.T,1e-8)


    Y1 = c.run_gen_mode(Win,W,Wout,np.array([data[trainLen]]),
                        x1[-1,:],testLen)

    errorLen = 500
    mse1 = sum(
        np.square(data[trainLen + 1:trainLen + errorLen + 1] -
                  Y1.T[0, 0:errorLen])) / errorLen
    print('MSE1 = ' + str(mse1))

def test_randomMatrix():
    trainLen = 10000
    testLen = 10000
    initLen = 100
    import random
    random.seed(42)
    data = np.array([np.array([random.randint(0,10)])for _ in range(trainLen+testLen+1)])[:,0]
    print(data.shape)

    inSize = outSize = 1
    resSize = 10000
    Yt = data[None, initLen + 1:trainLen + 1]
    np.random.seed(42)
    Win = np.random.rand(resSize, inSize) - 0.5

    W = m.generate_smatrix(resSize,resSize)
    m.scale_spectral_smatrix(W,in_place =True)


    print("run states")
    x1 = c.run_states(Win,W,data[:trainLen,None],
                      np.array([ 0 for _ in range(resSize)]))


    inputs = np.matrix(np.array([[x]for x in data[:trainLen]]))
    X1 = c.build_extended_states(inputs,x1,initLen)

    print("calculate Wout")
    Wout = t.ridge_reg(X1,Yt.T,1e-8)


    print("run gen mode")
    Y1 = c.run_gen_mode(Win,W,Wout,np.array([data[trainLen]]),
                        x1[-1,:],testLen)

    print("calculate mse")
    errorLen = 500
    print(data[trainLen + 1:trainLen + errorLen + 1].shape,
    Y1.T[0, 0:errorLen].shape)
    mse1 = sum(
        np.square(data[trainLen + 1:trainLen + errorLen + 1] -
                  Y1.T[0, 0:errorLen])) / errorLen
    print('MSE1 = ' + str(mse1))
