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
        rows, cols = self.shape
        dense = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for r, c, val in zip(self.row, self.col, self.data):
            dense[r][c] += val
        return dense

    def _add_impl(self, other: 'COOMatrix') -> 'COOMatrix':
        """Сложение COO матриц."""
        return COOMatrix(
            data=self.data + other.data,
            row=self.row + other.row,
            col=self.col + other.col,
            shape=self.shape
        )

    def _mul_impl(self, scalar: float) -> 'COOMatrix':
        """Умножение COO на скаляр."""
        new_data = [val * scalar for val in self.data]
        return COOMatrix(new_data, self.row, self.col, self.shape)

    def transpose(self) -> 'COOMatrix':
        """Транспонирование COO матрицы."""
        new_shape = (self.shape[1], self.shape[0])
        return COOMatrix(self.data, self.col, self.row, new_shape)

    def _matmul_impl(self, other: 'COOMatrix') -> 'COOMatrix':
        """Умножение COO матриц."""
        res_rows, res_cols, res_data = [], [], []
        temp_results = {}
        for r1, c1, v1 in zip(self.row, self.col, self.data):
            for r2, c2, v2 in zip(other.row, other.col, other.data):
                if c1 == r2:
                    idx = (r1, c2)
                    temp_results[idx] = temp_results.get(idx, 0.0) + v1 * v2
        for (r, c), val in temp_results.items():
            res_rows.append(r)
            res_cols.append(c)
            res_data.append(val)
        return COOMatrix(res_data, res_rows, res_cols, (self.shape[0], other.shape[1]))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        data, row_indices, col_indices = [], [], []
        for i in range(rows):
            for j in range(cols):
                val = dense_matrix[i][j]
                if val != 0:
                    data.append(float(val))
                    row_indices.append(i)
                    col_indices.append(j)
        return cls(data, row_indices, col_indices, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix
        sorted_indices = sorted(range(len(self.data)), key=lambda i: (self.col[i], self.row[i]))
        data = [self.data[i] for i in sorted_indices]
        indices = [self.row[i] for i in sorted_indices]
        indptr = [0] * (self.shape[1] + 1)
        for i in sorted_indices:
            indptr[self.col[i] + 1] += 1
        for i in range(len(indptr) - 1):
            indptr[i + 1] += indptr[i]
        return CSCMatrix(data, indices, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix
        sorted_indices = sorted(range(len(self.data)), key=lambda i: (self.row[i], self.col[i]))
        data = [self.data[i] for i in sorted_indices]
        indices = [self.col[i] for i in sorted_indices]
        indptr = [0] * (self.shape[0] + 1)
        for i in sorted_indices:
            indptr[self.row[i] + 1] += 1
        for i in range(len(indptr) - 1):
            indptr[i + 1] += indptr[i]
        return CSRMatrix(data, indices, indptr, self.shape)