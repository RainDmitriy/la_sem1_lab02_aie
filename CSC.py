from base import Matrix
from types import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix
from COO import COOMatrix
from CSR import CSRMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for j in range(cols):
            for k in range(self.indptr[j], self.indptr[j + 1]):
                dense[self.indices[k]][j] += self.data[k]
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        return self._to_coo()._add_impl(other)._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        data = [v * scalar for v in self.data]
        return CSCMatrix(data, self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Hint:
        Результат - в CSR формате (с теми же данными, но с интерпретацией строк как столбцов).
        """
        r, c = self.shape
        return CSRMatrix(self.data[:], self.indices[:], self.indptr[:], (c, r))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        a = self.to_dense()
        b = other.to_dense()
        n, m = self.shape[0], other.shape[1]
        k = self.shape[1]
        res = [[0.0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for t in range(k):
                if a[i][t] != 0:
                    for j in range(m):
                        res[i][j] += a[i][t] * b[t][j]
        return CSCMatrix.from_dense(res)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        data = []
        indices = []
        indptr = [0]
        for j in range(cols):
            for i in range(rows):
                if dense_matrix[i][j] != 0:
                    data.append(dense_matrix[i][j])
                    indices.append(i)
            indptr.append(len(data))
        return cls(data, indices, indptr, (rows, cols))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        return self._to_coo()._to_csr()

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        data = []
        row = []
        col = []
        rows, cols = self.shape
        for j in range(cols):
            for k in range(self.indptr[j], self.indptr[j + 1]):
                data.append(self.data[k])
                row.append(self.indices[k])
                col.append(j)
        return COOMatrix(data, row, col, self.shape)
