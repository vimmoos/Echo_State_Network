import pickle as p
import random as r
from collections import ChainMap
from os import listdir
from os.path import isfile, join
from pprint import pp, pprint

import matplotlib.pyplot as pl
import scipy.fft as f
import scipy.signal as s

import project.esn.transformer as t
import project.stats.metrics as met

path_max = "/home/vimmoos/NN/resources/esn/"

path_mar = "/home/pasta/Desktop/uni/secondYear/block-2b/NN/NN/project/fitter/dumps/"
path_s = "/home/sneha-lodha/Desktop/esn/"

experiment = lambda path: [f for f in listdir(path) if isfile(join(path, f))]

# data = (pic.load(open(f).__enter__()) for f in experiment)


def get_data(path):
    return [p.load(open(path + x, "rb")) for x in experiment(path)[:20]]


def apply_metrics(output, desired, raw_output):
    return {
        x.name: x.value(output, desired)()
        if x.name != "teacher_loss_nd" else x.value(raw_output, desired)()
        for x in list(met.Metrics)
    }


def apply_transformers(dict_, data_len):
    output = dict_["output"][:data_len]
    des = dict_["desired"][:data_len]
    return {
        trans.name: [{
            **{
                "param": (val := ((param * 2) / 10) + 0.2)
            },
            **apply_metrics(
                trans.value(val, t._identity)(output), des, output)
        } for param in range(5)]
        for trans in list(t.Transformers)
    }


def process_data(data, data_len):
    return ([{
        **apply_transformers(y, data_len),
        **y
    } for y in x] for x in data)


def remove_raw(data_gen, rkeys=["output", "desired", "input"]):
    for x in data_gen:
        removed = [x[idx].pop(key) for idx in range(len(x)) for key in rkeys]
        yield x
