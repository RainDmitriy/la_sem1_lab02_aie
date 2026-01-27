from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        if len(data) != len(row) or len(data) != len(col):
            raise ValueError("Длины data, row и col должны совпадать")

        for r, c in zip(row, col):
            if r < 0 or r >= shape[0] or c < 0 or c >= shape[1]:
                raise ValueError(f"Индекс ({r}, {c}) вне границ матрицы {shape}")

        self.data = data.copy()
        self.row = row.copy()
        self.col = col.copy()
        self.nnz = len(data)

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]

        for val, r, c in zip(self.data, self.row, self.col):
            dense[r][c] = val

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц."""
        dense_a = self.to_dense()
        dense_b = other.to_dense()

        rows, cols = self.shape
        result = [[0.0] * cols for _ in range(rows)]

        for i in range(rows):
            for j in range(cols):
                result[i][j] = dense_a[i][j] + dense_b[i][j]

        return COOMatrix.from_dense(result)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        if scalar == 0:
            return COOMatrix([], [], [], self.shape)

        new_data = [val * scalar for val in self.data]
        return COOMatrix(new_data, self.row.copy(), self.col.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        new_shape = (self.shape[1], self.shape[0])

        return COOMatrix(self.data.copy(), self.col.copy(), self.row.copy(), new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""
        from CSR import CSRMatrix
        csr_self = self._to_csr()

        if not isinstance(other, CSRMatrix):
            csr_other = CSRMatrix.from_dense(other.to_dense())
        else:
            csr_other = other

        result_csr = csr_self._matmul_impl(csr_other)

        return result_csr._to_coo()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0

        data = []
        row_indices = []
        col_indices = []

        for i in range(rows):
            for j in range(cols):
                val = dense_matrix[i][j]
                if val != 0:
                    data.append(val)
                    row_indices.append(i)
                    col_indices.append(j)

        return cls(data, row_indices, col_indices, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix

        sorted_indices = sorted(range(self.nnz), key=lambda i: (self.col[i], self.row[i]))

        data = []
        indices = []
        indptr = [0] * (self.shape[1] + 1)

        current_col = 0
        for idx in sorted_indices:
            col = self.col[idx]

            while current_col < col:
                indptr[current_col + 1] = len(data)
                current_col += 1

            data.append(self.data[idx])
            indices.append(self.row[idx])

        while current_col < self.shape[1]:
            indptr[current_col + 1] = len(data)
            current_col += 1

        return CSCMatrix(data, indices, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix

        sorted_indices = sorted(range(self.nnz), key=lambda i: (self.row[i], self.col[i]))

        data = []
        indices = []
        indptr = [0] * (self.shape[0] + 1)

        current_row = 0
        for idx in sorted_indices:
            row = self.row[idx]

            while current_row < row:
                indptr[current_row + 1] = len(data)
                current_row += 1

            data.append(self.data[idx])
            indices.append(self.col[idx])

        while current_row < self.shape[0]:
            indptr[current_row + 1] = len(data)
            current_row += 1

        return CSRMatrix(data, indices, indptr, self.shape)