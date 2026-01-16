from base import Matrix
from types import COOData, COORows, COOCols, Shape, DenseMatrix
from CSC import CSCMatrix

class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.row = row
        self.col = col

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for v, i, j in zip(self.data, self.row, self.col):
            dense[i][j] += v
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц."""
        other_coo = other._to_coo() if hasattr(other, "_to_coo") else other
        data = self.data + other_coo.data
        row = self.row + other_coo.row
        col = self.col + other_coo.col
        return COOMatrix(data, row, col, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        data = [v * scalar for v in self.data]
        return COOMatrix(data, self.row[:], self.col[:], self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        r, c = self.shape
        return COOMatrix(self.data[:], self.col[:], self.row[:], (c, r))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""
        a = self.to_dense()
        b = other.to_dense()
        n, m = self.shape[0], other.shape[1]
        k = self.shape[1]
        res = [[0.0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for t in range(k):
                for j in range(m):
                    res[i][j] += a[i][t] * b[t][j]
        return COOMatrix.from_dense(res)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        data = []
        row = []
        col = []
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        for i in range(rows):
            for j in range(cols):
                if dense_matrix[i][j] != 0:
                    data.append(dense_matrix[i][j])
                    row.append(i)
                    col.append(j)
        return cls(data, row, col, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        rows, cols = self.shape
        nnz = len(self.data)
        data = []
        indices = []
        indptr = [0] * (cols + 1)
        for j in self.col:
            indptr[j + 1] += 1
        for i in range(cols):
            indptr[i + 1] += indptr[i]
        cur = indptr[:]
        data = [0.0] * nnz
        indices = [0] * nnz
        for v, i, j in zip(self.data, self.row, self.col):
            pos = cur[j]
            data[pos] = v
            indices[pos] = i
            cur[j] += 1
        return CSCMatrix(data, indices, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        rows, cols = self.shape
        nnz = len(self.data)
        data = []
        indices = []
        indptr = [0] * (rows + 1)
        for i in self.row:
            indptr[i + 1] += 1
        for i in range(rows):
            indptr[i + 1] += indptr[i]
        cur = indptr[:]
        data = [0.0] * nnz
        indices = [0] * nnz
        for v, i, j in zip(self.data, self.row, self.col):
            pos = cur[i]
            data[pos] = v
            indices[pos] = j
            cur[i] += 1
        return CSRMatrix(data, indices, indptr, self.shape)
