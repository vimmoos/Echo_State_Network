import csv
import itertools as it
import multiprocessing as mp
import os
import pickle as p
import random as r
from collections import ChainMap
from os import listdir
from os.path import isfile, join
from pprint import pp, pprint

import matplotlib.pyplot as pl
import numpy as np
import scipy.fft as f
import scipy.signal as s

import project.esn.transformer as t
import project.esn.utils as u
import project.stats.metrics as met

path_resources = "/home/vimmoos/NN/resources/"

path_max = path_resources + "esn/final1/"

path_csv = path_resources + "esn_csv/final1/"

path_mar = "/home/pasta/Desktop/uni/secondYear/block-2b/NN/NN/project/fitter/dumps/"

path_s = "/home/sneha-lodha/Desktop/esn/"

experiment = lambda path: [
    join(path, f) for f in listdir(path) if isfile(join(path, f))
]

get_data = lambda paths: (p.load(open(f, "rb")) for f in paths)

get_experiment = u.comp(get_data, experiment)

squeeze_fs = [t._identity, t.my_sigm, np.tanh, t.sigmoid]


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
            "param": (val := ((param * 2) / 10) + 0.2),
            "squeeze_f":
            squeeze_f.__name__,
            **apply_metrics(trans.value(val, squeeze_f)(output), des, output)
        } for param in range(5) for squeeze_f in squeeze_fs]
        for trans in list(t.Transformers)
    }


def remove_raw(data_gen, rkeys=["output", "desired", "input"]):
    for x in data_gen:
        removed = [x[idx].pop(key) for idx in range(len(x)) for key in rkeys]
        yield x


@u.Comp(remove_raw)
def process_data(data, data_len, rkeys=None):
    return ([{
        **apply_transformers(y, data_len),
        **y
    } for y in x] for x in data)


def hget_metrics(metrics):
    return [x for x in metrics[0].keys() if x not in ["param", "squeeze_f"]]


def hget_params(params):
    return [
        x for x in params.keys()
        if x not in map(lambda x: x.name, list(t.Transformers))
    ]


def bget_metrics(metrics):
    return [metrics[0][x] for x in hget_metrics(metrics)]


def bget_tparam(metrics):
    return [metrics[x]["param"] for x in range(len(metrics))]


def bget_tsqueeze(metrics):
    return [metrics[x]["squeeze_f"] for x in range(len(squeeze_fs))]


def bget_params(params):
    return [params[x] for x in hget_params(params)]


first_t = list(t.Transformers)[0].name


def get_header(single_run):
    return [
        *hget_params(single_run[0]), "post_trans", "post_param",
        "post_squeeze", "metric", "metric_val"
        # *hget_metrics(single_run[0][first_t])
    ]


def get_body(single_run):
    return [[[[[
        *bget_params(redun), trans, t_param, t_squeeze, x, redun[trans][0][x]
    ] for x in hget_metrics(redun[trans])]
              for t_param in bget_tparam(redun[first_t])
              for t_squeeze in bget_tsqueeze(redun[first_t])]
             for trans in map(lambda x: x.name, list(t.Transformers))]
            for redun in single_run]


def my_flatten(lol):
    if not isinstance(lol[0][0], list):
        return lol
    acc = []
    for x in lol:
        acc += my_flatten(x)
    return acc


def to_csv(single_run, file):
    with open(file, "w") as f:
        writer = csv.writer(f)
        writer.writerows(
            [get_header(single_run), *my_flatten(get_body(single_run))])


def peek(iterable):
    try:
        first = next(iterable)
    except StopIteration:
        return None
    return first, it.chain([first], iterable)


def multiproc(func, args, workers, slicing=20):
    pool = mp.Pool(processes=workers)
    while True:
        res = pool.map(func, it.islice(args, slicing))
        if peek(args) is None:
            return res


def partition_data(graw_data, n):
    while (True):
        acc = []
        for _ in range(n):
            try:
                acc += [next(graw_data)]
            except StopIteration as e:
                return
        yield acc

    # raw_data = get_experiment(path)


def pre_post_proc(enumeration, cpath, data_len):
    (n, graw_data) = enumeration
    data = process_data(graw_data, data_len)
    for i, run in enumerate(data):
        to_csv(
            run, cpath + str(hash(str(os.getpid()))) + str(hash(str(i))) +
            str(hash(str(n))))
        print(f"dumped run {n}")


def partial_fun(enum):
    return pre_post_proc(enum, path_csv, 5000)


def cpre_post_proc(raw_data, size_group=4, workers=20):
    multiproc(partial_fun, enumerate(partition_data(raw_data, size_group)),
              workers)
