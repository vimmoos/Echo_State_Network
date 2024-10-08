import functools as ft
import random
from dataclasses import dataclass
from functools import partial
from math import isnan

import numpy as np
import scipy.stats as stats
from aenum import Enum, extend_enum
from scipy.spatial import distance as dist

import project.esn.utils as u


class Metrics(Enum):
    pass


nans = lambda li: [float("nan") if x == 0 else x for x in li]


@dataclass(init=True, repr=True, frozen=True)
class Metric():
    _metric: callable
    output: list
    target: list

    def __call__(self):
        ret = nans(self._metric(self.output, self.target))
        return np.nanmean(np.where(ret != 0, ret, np.nan))


def add_metric(func):
    inner = lambda *args, **kwargs: Metric(func, *args, **kwargs)
    name = func.__name__

    extend_enum(Metrics, name, inner)
    return func


@add_metric
def nmse(output, desired):
    return (sum(np.square(desired - output)) /
            sum(np.square(desired - np.mean(desired))))


teacher_log = np.vectorize(lambda out, teach: out if teach >= 1 else 1 - out)


@add_metric
def mse(output, desired):
    return sum(np.square(desired - output)) / len(output)


def teacher_loss_1d(output, teacher):
    return sum(teacher_log(output, teacher)) / len(output)


@add_metric
def teacher_loss_nd(output, teacher):
    return [
        teacher_loss_1d(output[:, dim], teacher[:, dim])
        for dim in range(output.shape[1])
    ]


@add_metric
def pearson_nd(output, teacher):
    return [
        stats.pearsonr(output[:, dim], teacher[:, dim])[0]
        for dim in range(output.shape[1])
    ]


@add_metric
def euclidian_distance(output, desired):
    return [
        dist.euclidean(output[:, dim], desired[:, dim])
        for dim in range(output.shape[1])
    ]


@add_metric
def manhattan_distance(output, desired):
    return [
        dist.cityblock(output[:, dim], desired[:, dim])
        for dim in range(output.shape[1])
    ]


@add_metric
def hamming_distance(output, desired):
    return [
        dist.hamming(output[:, dim], desired[:, dim])
        for dim in range(output.shape[1])
    ]


@add_metric
def np_cor(output, teacher):
    return (ft.reduce(lambda y, x: y + x, [
        np.correlate(output[:, dim], teacher[:, dim]).tolist()
        for dim in range(output.shape[1])
    ]) / np.sqrt(sum(output**2) * sum(teacher**2)))

@add_metric
def mse_cor(output,teacher):
    return np_cor(output,teacher) -  mse(output,teacher)
