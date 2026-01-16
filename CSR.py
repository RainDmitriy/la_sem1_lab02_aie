from base import Matrix
from types import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix
from COO import COOMatrix, CSCMatrix

class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for k in range(self.indptr[i], self.indptr[i + 1]):
                dense[i][self.indices[k]] += self.data[k]
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        return self._to_coo()._add_impl(other)._to_csr()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        data = [v * scalar for v in self.data]
        return CSRMatrix(data, self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Hint:
        Результат - в CSC формате (с теми же данными, но с интерпретацией столбцов как строк).
        """
        r, c = self.shape
        return CSCMatrix(self.data[:], self.indices[:], self.indptr[:], (c, r))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
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
        return CSRMatrix.from_dense(res)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        data = []
        indices = []
        indptr = [0]
        for i in range(rows):
            for j in range(cols):
                if dense_matrix[i][j] != 0:
                    data.append(dense_matrix[i][j])
                    indices.append(j)
            indptr.append(len(data))
        return cls(data, indices, indptr, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSRMatrix в CSCMatrix.
        """
        return self._to_coo()._to_csc()

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSRMatrix в COOMatrix.
        """
        data = []
        row = []
        col = []
        rows, _ = self.shape
        for i in range(rows):
            for k in range(self.indptr[i], self.indptr[i + 1]):
                data.append(self.data[k])
                row.append(i)
                col.append(self.indices[k])
        return COOMatrix(data, row, col, self.shape)
