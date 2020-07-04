import random
from dataclasses import dataclass
import numpy as np
from aenum import extend_enum, Enum
import project.esn.utils as u
from functools import partial
import functools as ft
from scipy.spatial.distance import cdist

class Metrics(Enum):
    pass

@dataclass(init=True, repr=True, frozen = True)
class Metric():
    _metric: callable
    output: list
    target: list

    def __call__(self):
        return self._metric(self.output, self.target)

def add_metric(func):
    inner = lambda *args, **kwargs: Metric(func, *args, **kwargs)
    name = func.__name__

    extend_enum(Metrics, name, inner)
def nmse(output, desired):
    return (sum(np.square(desired - output)) /
            sum(np.square(desired - np.mean(desired))))


teacher_log = np.vectorize(lambda out, teach: out if teach >= 1 else 1 - out)

@add_metric
def mse(output, desired):
    return sum(np.square(desired - output)) / len(output)

@add_metric
def teacher_loss_1d(output, teacher):
    return sum(teacher_log(output, teacher)) / len(output)

@add_metric
def teacher_loss_nd(output, teacher):
    return [
        teacher_loss_1d(output[:, dim], teacher[:, dim])
        for dim in range(output.shape[1])
    ]

@add_metric
def euclidian_distance(output, desired):
    return cdist(output, desired, 'euclidean')

@add_metric
def manhattan_distance(output, desired):
    return cdist(output, desired, 'cityblock')

@add_metric
def hamming_distance(output, desired):
    return cdist(output, desired, 'hamming')

@add_metric
def np_cor(output, teacher):
    return (ft.reduce(lambda y, x: y + x, [
        np.correlate(output[:, dim], teacher[:, dim]).tolist()
        for dim in range(output.shape[1])
    ]) / np.sqrt(sum(output**2) * sum(teacher**2)))
