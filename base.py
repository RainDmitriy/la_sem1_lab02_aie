from abc import ABC, abstractmethod
from type import DenseMatrix, Shape

class Matrix(ABC):
    def __init__(self, shape: Shape):
        self.shape = shape

    @abstractmethod
    def to_dense(self) -> DenseMatrix:
        """Трансформация в формат вложенных списков."""
        pass

    def __add__(self, other: 'Matrix') -> 'Matrix':
        if self.shape != other.shape:
            raise ValueError(f"Shapes {self.shape} and {other.shape} are inconsistent")
        return self._add_impl(other)

    @abstractmethod
    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        pass

    def __mul__(self, val: float) -> 'Matrix':
        return self._mul_impl(float(val))

    def __rmul__(self, val: float) -> 'Matrix':
        return self.__mul__(val)

    @abstractmethod
    def _mul_impl(self, scalar: float) -> 'Matrix':
        pass

    @abstractmethod
    def transpose(self) -> 'Matrix':
        pass

    def __matmul__(self, other: 'Matrix') -> 'Matrix':
        if self.shape[1] != other.shape[0]:
            raise ValueError("Incompatible dimensions for matmul")
        return self._matmul_impl(other)

    @abstractmethod
    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        pass
