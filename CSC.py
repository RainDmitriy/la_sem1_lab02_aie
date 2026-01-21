from .base import Matrix
from .types import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        sr, sc = shape
        if len(indptr) != sc + 1:
            raise ValueError("indptr должен иметь длину cols + 1")
        if len(data) != len(indices):
            raise ValueError("data и indices должны быть одинаковой длины")
        if indptr[0] != 0:
            raise ValueError("indptr[0] должен быть 0")
        if indptr[-1] != len(data):
            raise ValueError("indptr[-1] должен совпадать с len(data)")
        for k in indices:
            if k < 0 or k >= sr:
                raise ValueError("indices индекс вне диапазона")

        self.data = [float(x) for x in data]
        self.indices = [int(x) for x in indices]
        self.indptr = [int(x) for x in indptr]

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу"""
        sr, sc = self.shape
        dm = [[0.0 for _ in range(sc)] for _ in range(sr)]
        for c in range(sc):
            s = self.indptr[c]
            e = self.indptr[c + 1]
            for p in range(s, e):
                r = self.indices[p]
                dm[r][c] += self.data[p]
        return dm

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц"""
        coo = self._to_coo()
        res = coo._add_impl(other)
        return res._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр"""
        data = [v * float(scalar) for v in self.data]
        return CSCMatrix(data, list(self.indices), list(self.indptr), self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы
        """
        from .CSR import CSRMatrix

        sr, sc = self.shape
        return CSRMatrix(list(self.data), list(self.indices), list(self.indptr), (sc, sr))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц"""
        a = self._to_csr()
        c = a._matmul_impl(other)
        if hasattr(c, "_to_csc"):
            return c._to_csc()
        from .COO import COOMatrix
        return COOMatrix.from_dense(c.to_dense())._to_csc()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы"""
        sr = len(dense_matrix)
        sc = len(dense_matrix[0]) if sr > 0 else 0
        for r in range(sr):
            if len(dense_matrix[r]) != sc:
                raise ValueError("Плотная матрица должна быть прямоугольной")

        data: list[float] = []
        i: list[int] = []
        indptr: list[int] = [0]

        for c in range(sc):
            for r in range(sr):
                d = float(dense_matrix[r][c])
                if d != 0.0:
                    data.append(d)
                    i.append(r)
            indptr.append(len(data))

        return cls(data, i, indptr, (sr, sc))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix
        """
        return self._to_coo()._to_csr()

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSCMatrix в COOMatrix
        """
        from .COO import COOMatrix

        sr, sc = self.shape
        data: list[float] = []
        row: list[int] = []
        col: list[int] = []
        for c in range(sc):
            s = self.indptr[c]
            e = self.indptr[c + 1]
            for p in range(s, e):
                row.append(self.indices[p])
                col.append(c)
                data.append(self.data[p])
        return COOMatrix(data, row, col, self.shape)
