from project.esn.utils import register_methods, callable_attr

import abc

matrix_pow = lambda self, other: self.scalar_product(other)

matrix_mul = lambda self, other: self.dot_product(other)

matrix_imul = lambda self, other: self.__mul__(other)

matrix_ipow = lambda self, other: self.__pow__(other)

matrix_inv = lambda self: self.inverse()

matrix_add = lambda self, other: self.add(other)


@register_methods({
    "__pow__": matrix_pow,
    "__ipow__": matrix_ipow,
    "__mul__": matrix_mul,
    "__imul__": matrix_imul,
    "__invert__": matrix_inv,
    "__add__": matrix_add,
})
class Matrix(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasscheck__(cls, subclass):
        return (callable_attr(subclass, "shape")
                and callable_attr(subclass, "dot_product")
                and callable_attr(subclass, "scalar_product")
                and callable_attr(subclass, "eigenvals")
                and callable_attr(subclass, "inverse")
                and callable_attr(subclass, "T")
                and callable_attr(subclass, "add"))
