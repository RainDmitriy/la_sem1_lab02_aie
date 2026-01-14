from base import Matrix
from types import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix
from CSR import CSRMatrix
from COO import COOMatrix

class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = list(data)
        self.indices = list(indices)
        self.indptr = list(indptr)

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу."""
        n, m = self.shape
        dense = [[0.0 for _ in range(m)] for _ in range(n)]
        for j in range(m):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for k in range(start, end):
                i = self.indices[k]
                dense[i][j] += self.data[k]
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        if not isinstance(other, CSCMatrix):
            if hasattr(other, "_to_csc"):
                other = other._to_csc()
            else:
                other = CSCMatrix.from_dense(other.to_dense())

        coo = self._to_coo()
        coo2 = other._to_coo()
        return coo._add_impl(coo2)._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        if scalar == 0:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)
        data = [v * scalar for v in self.data]
        return CSCMatrix(data, list(self.indices), list(self.indptr), self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Hint:
        Результат - в CSR формате (с теми же данными, но с интерпретацией строк как столбцов).
        """
        n, m = self.shape
        return CSRMatrix(list(self.data), list(self.indices), list(self.indptr), (m, n))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        from COO import COOMatrix
        a = self._to_coo()
        if isinstance(other, COOMatrix):
            b = other
        elif hasattr(other, "_to_coo"):
            b = other._to_coo()
        else:
            b = COOMatrix.from_dense(other.to_dense())
        return a._matmul_impl(b)._to_csc()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        n = len(dense_matrix)
        m = len(dense_matrix[0]) if n > 0 else 0
        data, indices = [], []
        indptr = [0]
        count = 0
        for j in range(m):
            for i in range(n):
                v = dense_matrix[i][j]
                if v != 0:
                    data.append(v)
                    indices.append(i)
                    count += 1
            indptr.append(count)
        return cls(data, indices, indptr, (n, m))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        return self._to_coo()._to_csr()

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        n, m = self.shape
        row, col, data = [], [], []
        for j in range(m):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for k in range(start, end):
                row.append(self.indices[k])
                col.append(j)
                data.append(self.data[k])
        return COOMatrix(data, row, col, (n, m))