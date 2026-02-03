from base import Matrix
from types import COOData, COORows, COOCols, Shape, DenseMatrix


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.row = row
        self.col = col

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу."""
        dense = [[0.0 for _ in range(self.shape[1])] for _ in range(self.shape[0])]
        for r, c, v in zip(self.row, self.col, self.data):
            dense[r][c] += v
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц."""
        return COOMatrix(self.data + other.data, self.row + other.row, self.col + other.col, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        return COOMatrix([val * scalar for val in self.data], self.row[:], self.col[:], self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        return COOMatrix(self.data[:], self.col[:], self.row[:], (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""
        return self._to_csr()._matmul_impl(other)._to_coo()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        data, rows, cols = [], [], []
        for r, row_vals in enumerate(dense_matrix):
            for c, val in enumerate(row_vals):
                if val != 0:
                    data.append(val)
                    rows.append(r)
                    cols.append(c)
        return cls(data, rows, cols, (len(dense_matrix), len(dense_matrix[0])))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix
        triplets = sorted(zip(self.col, self.row, self.data))
        c_sorted, r_sorted, d_sorted = zip(*triplets) if triplets else ([], [], [])
        indptr = [0] * (self.shape[1] + 1)
        for c in c_sorted:
            indptr[c + 1] += 1
        for i in range(self.shape[1]):
            indptr[i + 1] += indptr[i]
        return CSCMatrix(list(d_sorted), list(r_sorted), indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix
        triplets = sorted(zip(self.row, self.col, self.data))
        r_sorted, c_sorted, d_sorted = zip(*triplets) if triplets else ([], [], [])
        indptr = [0] * (self.shape[0] + 1)
        for r in r_sorted:
            indptr[r + 1] += 1
        for i in range(self.shape[0]):
            indptr[i + 1] += indptr[i]
        return CSRMatrix(list(d_sorted), list(c_sorted), indptr, self.shape)

    def _to_coo(self) -> 'COOMatrix':
        return self
