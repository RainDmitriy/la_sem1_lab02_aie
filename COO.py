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
        dense = []
        rows, cols = self.shape
        for _ in range(rows):
            new_row = [0.0] * cols
            dense.append(new_row)
        nnz_count = len(self.data)
        for i in range(nnz_count):
            row, col = self.row[i], self.col[i]
            value = self.data[i]
            dense[row][col] += value
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц."""
        new_data = self.data + other.data
        new_rows = self.row + other.row
        new_cols = self.col + other.col
        return COOMatrix(new_data, new_rows, new_cols, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        new_data = []
        for val in self.data:
            new_data.append(val * scalar)
        return COOMatrix(new_data, list(self.row), list(self.col), self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        new_shape = (self.shape[1], self.shape[0])
        return COOMatrix(list(self.data), list(self.col), list(self.row), new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""
        res_data = []
        res_row = []
        res_col = []
        for i in range(len(self.data)):
            val1 = self.data[i]
            row1 = self.row[i]
            col1 = self.col[i]

            for j in range(len(other.data)):
                val2 = other.data[j]
                row2 = other.row[j]
                col2 = other.col[j]

                if col1 == row2:
                    res_data.append(val1 * val2)
                    res_row.append(row1)
                    res_col.append(col2)
        return COOMatrix(res_data, res_row, res_col, (self.shape[0], other.shape[1]))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        data = []
        rows = []
        cols = []
        num_rows = len(dense_matrix)
        num_cols = len(dense_matrix[0])

        for i in range(num_rows):
            for j in range(num_cols):
                value = dense_matrix[i][j]
                if value != 0:
                    data.append(value)
                    rows.append(i)
                    cols.append(j)
        return cls(data, rows, cols, (num_rows, num_cols))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix
        dense = self.to_dense()
        return CSCMatrix.from_dense(dense)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix
        dense = self.to_dense()
        return CSRMatrix.from_dense(dense)

    def _to_coo(self) -> 'COOMatrix':
        return self
