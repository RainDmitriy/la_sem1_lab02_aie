from abc import ABC, abstractmethod
from type import DenseMatrix, Shape


class Matrix(ABC):
    def __init__(self, shape: Shape):
        self.shape = shape
        

    @abstractmethod
    def to_dense(self) -> DenseMatrix:
        data = getattr(self, "data", None)
        if data is None:
            raise AttributeError()

        r, c = self.shape

        if r == 0:
            return []
        if c == 0:
            return [[] for _ in range(r)]

        if len(data) != r or any(len(row) != c for row in data):
            raise ValueError("Данные матрицы не соответствуют shape")
        
        return [row[:] for row in data]
    
    @classmethod
    @abstractmethod
    def from_dense(cls, data: DenseMatrix) -> "Matrix":
        raise NotImplementedError
    

    def __add__(self, other: 'Matrix') -> 'Matrix':
        """Сложение матриц."""
        if self.shape != other.shape:
            raise ValueError("Размерности матриц не совпадают")
        return self._add_impl(other)

    @abstractmethod
    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        a = self.to_dense()
        b = other.to_dense()
        r, c = self.shape

        if r == 0:
            res: DenseMatrix = []
        elif c == 0:
            res = [[] for _ in range(r)]
        else:
            res = [[0.0] * c for _ in range(r)]
            for i in range(r):
                ai = a[i]
                bi = b[i]
                ri = res[i]
                for j in range(c):
                    ri[j] = ai[j] + bi[j]

        return type(self).from_dense(res)
    

    def __mul__(self, scalar: float) -> 'Matrix':
        """Умножение на скаляр."""
        return self._mul_impl(scalar)
    

    @abstractmethod
    def _mul_impl(self, scalar: float) -> 'Matrix':
        a = self.to_dense()
        r, c = self.shape
        scalar = float(scalar)

        if r == 0:
            res: DenseMatrix = []
        elif c == 0:
            res = [[] for _ in range(r)]
        else:
            res = [[0.0] * c for _ in range(r)]
            for i in range(r):
                ai = a[i]
                ri = res[i]
                for j in range(c):
                    ri[j] = ai[j] * scalar

        return type(self).from_dense(res)
    

    def __rmul__(self, scalar: float) -> 'Matrix':
        """Обратное умножение на скаляр."""
        return self.__mul__(scalar)
    

    @abstractmethod
    def transpose(self) -> 'Matrix':
        a = self.to_dense()
        r, c = self.shape

        if c == 0:
            res: DenseMatrix = []
        elif r == 0:
            res = [[] for _ in range(c)]
        else:
            res = [[0.0] * r for _ in range(c)]
            for i in range(r):
                ai = a[i]
                for j in range(c):
                    res[j][i] = ai[j]

        return type(self).from_dense(res)
    
    
    def __matmul__(self, other: 'Matrix') -> 'Matrix':
        """Умножение матриц."""
        if self.shape[1] != other.shape[0]:
            raise ValueError("Несовместимые размерности для умножения")
        return self._matmul_impl(other)
    

    @abstractmethod
    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        a = self.to_dense()
        b = other.to_dense()

        r, k = self.shape
        _, c = other.shape

        if r == 0:
            res: DenseMatrix = []
        elif c == 0:
            res = [[] for _ in range(r)]
        elif k == 0:
            res = [[0.0] * c for _ in range(r)]
        else:
            res = [[0.0] * c for _ in range(r)]
            for i in range(r):
                ai = a[i]
                ri = res[i]
                for t in range(k):
                    a_it = ai[t]
                    if a_it == 0.0:
                        continue
                    bt = b[t]
                    for j in range(c):
                        ri[j] += a_it * bt[j]

        return type(self).from_dense(res)
    
    
class Dense(Matrix):
    def __init__(self, shape: Shape, data: DenseMatrix):
        super().__init__(shape)
        self.data = data

    @classmethod
    def from_dense(cls, data: DenseMatrix) -> "Dense":
        r = len(data)
        if r == 0:
            return cls((0, 0), [])
        c = len(data[0])
        if any(len(row) != c for row in data):
            raise ValueError("Плотная матрица должна быть прямоугольной")
        return cls((r, c), data)

    def to_dense(self): return super().to_dense()
    def _add_impl(self, other): return super()._add_impl(other)
    def _mul_impl(self, scalar): return super()._mul_impl(scalar)
    def transpose(self): return super().transpose()
    def _matmul_impl(self, other): return super()._matmul_impl(other)
    

