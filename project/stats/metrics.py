import random
import numpy as np
import scipy.stats as stats


def mse(output, desired):
    return sum(np.square(desired - output)) / len(output)




teacher_log = np.vectorize(lambda out, teach: out
                           if teach >= 1 else 1 - out)


def teacher_loss_1d(output, teacher):
    return sum(teacher_log(output,teacher)) / len(output)


def teacher_loss_nd(output, teacher):
    return [
        teacher_loss_1d(output[:, dim], teacher[:, dim])
        for dim in range(output.shape[1])
    ]

def pearson_nd(output,teacher):
    return [stats.pearsonr(output[:,dim],teacher[:,dim]) for dim in range(output.shape[1])]



# if (__name__ == "__main__"):
#     print(teacher_loss_nd(out, teacher, len(teacher)))
