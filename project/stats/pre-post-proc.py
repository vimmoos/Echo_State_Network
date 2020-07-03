import pickle as p
from os import listdir
from os.path import isfile, join
import project.esn.transformer as t
from collections import ChainMap

path = "/home/vimmoos/NN/resources/esn/"

experiment = (f for f in listdir(path) if isfile(join(path, f)))

# data = (pic.load(open(f).__enter__()) for f in experiment)


def get_data():
    return [p.load(open(path + x, "rb")) for x in experiment]


def process_data(data):
    return [[{
        **{
            trans.name: [{
                **{
                    "param": (val := ((param * 2) / 10) + 0.2)
                },
                **{
                    "output": trans.value(val, t._identity)(y["output"])
                }
            } for param in range(5)]
            for trans in list(t.Transformers)
        },
        **{k: v
           for k, v in y.items() if k != "output"}
    } for y in x] for x in data]


import matplotlib.pyplot as pl
import scipy.signal as s
import scipy.fft as f

import random as r


def _show(fun, data, data_len, max_len, transformer):
    r.shuffle(data)
    for i, x in enumerate(data):
        if i >= max_len:
            pl.show()
            return
        pl.figure(i)
        fun(x, data, data_len, transformer)
        pl.legend(range(8))


showable = lambda fun: lambda *args, **kwargs: _show(fun, *args, **kwargs)


def get_title(current):
    return " ".join([
        f"{k}={v}" for k, v in current.items() if k not in
        ["input", "desired", *[x.name for x in list(t.Transformers)]]
    ])


@showable
def output(
    current,
    data,
    data_len,
    transformer,
):
    pl.plot(current[0][transformer][0]["output"][:data_len])
    pl.title(get_title(current[0]) + " output")


@showable
def correlation(current, data, data_len, transformer):
    des = data[0][0]["desired"][:data_len]
    out = current[0][transformer][0]["output"][:data_len]
    pl.plot(s.correlate(out, des))
    pl.title(get_title(current[0]) + " correlation")


@showable
def fft(current, data, data_len, transformer):
    pl.plot(f.fftn(current[0][transformer][0]["output"][:data_len]))
    pl.title(get_title(current[0]) + " fft")


# def  test():
#     a = f.fftn(process_data[0][0]["pow_prob"][0]["output"][:500, 0])
#     c = f.fftn(process_data[0][0]["sig_prob"][0]["output"][:500, 0])
#     d = f.fftn(process_data[0][0]["threshold"][0]["output"][:500, 0])
#     e = f.fftn(process_data[0][0]["identity"][0]["output"][:500, 0])
#     b = f.fftn(process_data[0][0]["desired"][:500, 0])

#     pl.figure(0)
#     pl.plot(f.fftn(process_data[0][0]["desired"][:500]))
#     pl.figure(1)
#     pl.plot(f.fftn(process_data[0][0]["sig_prob"][0]["output"][:500]))
#     pl.figure(2)
#     pl.plot(f.fftn(process_data[0][0]["threshold"][0]["output"][:500]))
#     pl.show()

#     pl.figure(0)
#     pl.plot(np.log(np.abs(scipy.fft.fftshift(b))**2))
#     pl.figure(1)
#     pl.plot(np.log(np.abs(scipy.fft.fftshift(a))**2))
#     pl.figure(2)
#     pl.plot(np.log(np.abs(scipy.fft.fftshift(c))**2))
#     pl.figure(3)
#     pl.plot(np.log(np.abs(scipy.fft.fftshift(d))**2))
#     pl.figure(4)
#     pl.plot(np.log(np.abs(scipy.fft.fftshift(e))**2))

#     pl.show()
