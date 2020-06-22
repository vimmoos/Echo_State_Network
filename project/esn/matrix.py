import numpy as np
from scipy import sparse, stats, linalg
from project.esn.utils import mydataclass, register_methods, Comp, factories
from project.esn.imatrix import Matrix

shape = lambda self: self.matrix.shape

scalar_product = lambda self, other: self.matrix * other

add = lambda self, other: self.matrix + other

dot_product = lambda self, other: self.matrix.dot(other)

transpose = lambda self: self.matrix.T


@mydataclass(init=True, repr=True, check=True)
@factories("add", "scalar_product", "inverse", "T", "dot_product")
@register_methods({
    "dot_product": dot_product,
    "shape": shape,
    "add": add,
    "scalar_product": scalar_product,
    "T": transpose
})
class Dense_matrix(Matrix):
    matrix: np.ndarray = np.eye(0)

    def eigenvals(self):
        return linalg.eigs(self.matrix)

    def inverse(self):
        return linalg.inv(self.matrix)


@mydataclass(init=True, repr=True, check=True)
@factories("add", "scalar_product", "inverse", "T")
@register_methods({
    "dot_product": dot_product,
    "shape": shape,
    "add": add,
    "scalar_product": scalar_product,
    "T": transpose
})
class Sparse_matrix(Matrix):
    matrix: sparse.issparse = sparse.eye(0)

    def eigenvals(self):
        return sparse.linalg.eigvals(self.matrix)

    def inverse(self):
        return sparse.linalg.inv(self.matrix)


''' generator for scipy and np matrix '''


@Comp(Sparse_matrix)
def generate_smatrix(m, n, density=0.5, bound=0.5, **kwargs):
    np.random.seed(42)
    smatrix = sparse.random(m,
                            n,
                            density=density,
                            format="csr",
                            data_rvs=stats.uniform(-bound, 1).rvs)
    return smatrix


@Comp(Dense_matrix)
def generate_rmatrix(m, n, bound=0.5, **kwargs):
    return np.random.rand(m, n) - bound


@Comp(Dense_matrix)
def eye(shape):
    return np.eye(shape)


@Comp(Dense_matrix)
def zeros(shape):
    return np.zeros(shape)


def scale_spectral_smatrix(matrix: Matrix,
                           spectral_radius=0.9,
                           in_place=False):
    if not in_place:
        return matrix**(spectral_radius / max(abs(matrix.eigenvals()[0])))
    matrix **= (spectral_radius / max(abs(matrix.eigenvals()[0])))


@mydataclass(init=True, repr=True, check=True)
class Esn_matrixs():
    W_in: Matrix
    W_res: Matrix
    W_feb: Matrix = zeros((0, 0))
    W_out: Matrix = zeros((0, 0))
    spectral_radius: float = 0.9

    def __post_init__(self):
        scale_spectral_smatrix(self.W_res,
                               spectral_radius=self.spectral_radius,
                               in_place=True)
