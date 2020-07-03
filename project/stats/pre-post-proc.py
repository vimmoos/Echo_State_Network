import pickle as p
from os import listdir
from os.path import isfile, join
import project.esn.transformer as t
from collections import ChainMap

path = "/home/vimmoos/NN/resources/esn/"

experiment = [f for f in listdir(path) if isfile(join(path, f))]

# data = (pic.load(open(f).__enter__()) for f in experiment)


def get_data():
    return [p.load(open(path + x, "rb")) for x in experiment[:20]]


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
import numpy as np

import random as r


def _show(fun, data, data_len, max_len, transformer, desired=False):
    r.shuffle(data)
    for i, x in enumerate(data):
        if i >= max_len:
            break
        pl.figure(i)
        pl.plot(fun(x, data, data_len, transformer))
        pl.title(get_title(x[0]) + f" {fun.__name__}")
        pl.legend(range(9))
    if desired:
        pl.figure(max_len)
        pl.plot(fun(None, data, data_len, transformer))
        pl.title("desired" + f" {fun.__name__}")
        pl.legend(range(9))
    pl.show()


showable = lambda desired=False: lambda fun: lambda *args, **kwargs: _show(
    fun, *args, **kwargs, desired=desired)


def get_title(current):
    return " ".join([
        f"{k}={v}" for k, v in current.items() if k not in
        ["input", "desired", *[x.name for x in list(t.Transformers)]]
    ])


@showable(True)
def output(
           current,
           data,
           data_len,
           transformer,
           ):
    return current[0][transformer][0][
        "output"][:data_len] if current is not None else data[0][0][
            "desired"][:data_len]


@showable()
def correlation(current, data, data_len, transformer):
    des = data[0][0]["desired"][:data_len]
    out = current[0][transformer][0][
        "output"][:data_len] if current is not None else des
    return s.correlate(out, des)


@showable()
def fft(current, data, data_len, transformer):
    current = current[0][transformer][0][
        "output"][:data_len] if current is not None else data[0][0][
            "desired"][:data_len]
    return f.fftn(current)


@showable(True)
def log_fft(current, data, data_len, transformer):
    tmp = current[0][transformer][0][
        "output"][:data_len] if current is not None else data[0][0][
            "desired"][:data_len]
    return np.log(np.abs(f.fftshift(f.fftn(tmp)))**2)
