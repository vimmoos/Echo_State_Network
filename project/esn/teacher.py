import random

import numpy as np

random.seed(42)

rand_range = lambda r: [random.random() for _ in range(r)]

out = np.array([rand_range(4) for _ in range(4)])
teacher = np.array([[0, 1, 1, 1, 0], [0, 1, 0, 0, 0], [1, 1, 1, 0, 0],
                    [0, 1, 0, 0, 0]])

out1 = np.array([0.5, 0.7, 0.99, 0.3, 0.786])
teacher1 = np.array([0, 1, 0, 0, 1])


def _mse1d(output, desired, error_len):
    return (sum(np.square(desired[:error_len] - output[:error_len])) /
            error_len)


def _mse_nd(output, desired, error_len):
    return [
        _mse1d(output[:, x], desired[:, x],error_len)
        for x in range(output.shape[1])
    ]


teacher_log = np.vectorize(lambda out, teach: np.log(out)
                           if teach >= 1 else np.log(1 - out))


def teacher_loss_1d(output,
                    teacher,
                    precision_len):
    return sum([
        teacher_log(out, teach) for out, teach in zip(output[:precision_len],
                                                      teacher[:precision_len])
    ]) / precision_len


def teacher_loss_nd(output, teacher, precision_len):
    return [
        teacher_loss_1d(output[:, dim], teacher[:, dim], precision_len)
        for dim in range(output.shape[1])
    ]


# if (__name__ == "__main__"):
#     print(teacher_loss_nd(out, teacher, len(teacher)))
