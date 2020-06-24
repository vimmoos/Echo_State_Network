import numpy as np


def n_softmax(n: int):
    def inner(el: np.array):
        norm_output = np.exp(el) / sum(np.exp(el))
        highest_idx = np.argpartition(norm_output, -n)[-n:]
        return np.array([
            1 if idx in highest_idx else 0 for idx, _ in enumerate(norm_output)
        ])

    return inner


user_threshold = lambda t: np.vectorize(lambda el: 1 if el > t else 0)

# test = np.array([1, 2, 3, 4, 0.67, 0.09, 78])

# func = user_threshold(0.78)
# print(func(test))

# f = n_softmax(4)
# print(f(test))
