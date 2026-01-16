from abc import ABC
from types import DenseMatrix, Shape
from COO import COOMatrix


class Matrix(ABC):
    def __init__(self, shape: Shape):
        self.shape = shape

    def to_dense(self) -> DenseMatrix:
        """Преобразует разреженную матрицу в плотную."""
        raise NotImplementedError

    def __add__(self, other: 'Matrix') -> 'Matrix':
        """Сложение матриц."""
        if self.shape != other.shape:
            raise ValueError("Размерности матриц не совпадают")
        return self._add_impl(other)

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Реализация сложения с другой матрицей."""
        a = self.to_dense()
        b = other.to_dense()
        rows, cols = self.shape
        res = [[a[i][j] + b[i][j] for j in range(cols)] for i in range(rows)]
        return COOMatrix.from_dense(res)

    def __mul__(self, scalar: float) -> 'Matrix':
        """Умножение на скаляр."""
        return self._mul_impl(scalar)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Реализация умножения на скаляр."""
        a = self.to_dense()
        rows, cols = self.shape
        res = [[a[i][j] * scalar for j in range(cols)] for i in range(rows)]
        return COOMatrix.from_dense(res)

    def __rmul__(self, scalar: float) -> 'Matrix':
        """Обратное умножение на скаляр."""
        return self.__mul__(scalar)

    def transpose(self) -> 'Matrix':
        """Транспонирование матрицы."""
        a = self.to_dense()
        rows, cols = self.shape
        res = [[a[i][j] for i in range(rows)] for j in range(cols)]
        return COOMatrix.from_dense(res)

    def __matmul__(self, other: 'Matrix') -> 'Matrix':
        """Умножение матриц."""
        if self.shape[1] != other.shape[0]:
            raise ValueError("Несовместимые размерности для умножения")
        return self._matmul_impl(other)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Реализация умножения матриц."""
        a = self.to_dense()
        b = other.to_dense()
        n = self.shape[0]
        m = other.shape[1]
        k = self.shape[1]
        res = [[0.0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for t in range(k):
                if a[i][t] != 0:
                    for j in range(m):
                        res[i][j] += a[i][t] * b[t][j]
        return COOMatrix.from_dense(res)
