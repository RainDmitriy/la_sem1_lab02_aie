from abc import ABC, abstractmethod
from matrix_types import DenseMatrix, Shape


class Matrix(ABC):
    def __init__(self, shape: Shape):
        self.shape = shape

    @abstractmethod
    def to_dense(self) -> DenseMatrix:
        """Преобразует разреженную матрицу в плотную."""
        pass

    def __add__(self, other: 'Matrix') -> 'Matrix':
        """Сложение матриц."""
        if self.shape != other.shape:
            raise ValueError("Размерности матриц не совпадают")
        return self._add_impl(other)

    @abstractmethod
    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Реализация сложения с другой матрицей."""
        a = self.to_dense()
        b = other.to_dense()
        n, m = self.shape
        result = [[0 for _ in range(m)] for _ in range(n)]

        for i in range(n):
            for j in range(m):
                result[i][j] = a[i][j] + b[i][j]

        return DenseMatrix(result)

    def __mul__(self, scalar: float) -> 'Matrix':
        """Умножение на скаляр."""
        return self._mul_impl(scalar)

    @abstractmethod
    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Реализация умножения на скаляр."""
        n, m = self.shape
        result = [[0 for _ in range(m)] for _ in range(n)]

        for i in range(n):
            for j in range(m):
                result[i][j] = self[i][j] * scalar

        return DenseMatrix(result)    

    def __rmul__(self, scalar: float) -> 'Matrix':
        """Обратное умножение на скаляр."""
        return self.__mul__(scalar)

    @abstractmethod
    def transpose(self) -> 'Matrix':
        """Транспонирование матрицы."""
        n, m = self.shape
        result = [[0 for _ in range(m)] for _ in range(n)]

        for i in range(n):
            for j in range(m):
                result[i][j] = self[j][i]

        return DenseMatrix(result)    

    def __matmul__(self, other: 'Matrix') -> 'Matrix':
        """Умножение матриц."""
        if self.shape[1] != other.shape[0]:
            raise ValueError("Несовместимые размерности для умножения")
        return self._matmul_impl(other)

    @abstractmethod
    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Реализация умножения матриц."""
        a = self.to_dense()
        b = other.to_dense()

        n, m = self.shape
        _, k = other.shape

        result = [[0 for _ in range(k)] for _ in range(n)]

        for i in range(n):
            for j in range(k):
                for t in range(m):
                    result[i][j] += a[i][t] * b[t][j]

        return DenseMatrix(result)