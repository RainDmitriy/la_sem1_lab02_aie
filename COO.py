from base import Matrix
from typing import List, Tuple

DenseMatrix = List[List[float]]
Shape = Tuple[int, int]


class COOMatrix(Matrix):
    def __init__(self, data: List[float], rows: List[int], cols: List[int], shape: Shape):
        super().__init__(shape)
        if len(data) != len(rows) or len(data) != len(cols):
            raise ValueError("Длины data, rows и cols должны совпадать")

        for r, c in zip(rows, cols):
            if r < 0 or r >= shape[0] or c < 0 or c >= shape[1]:
                raise ValueError(f"Индекс ({r}, {c}) вне границ матрицы {shape}")

        self.data = data.copy()
        self.rows = rows.copy()
        self.cols = cols.copy()
        self.nnz = len(data)

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]

        for val, r, c in zip(self.data, self.rows, self.cols):
            dense[r][c] = val

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц."""
        from CSC import CSCMatrix
        from CSR import CSRMatrix

        if isinstance(other, (COOMatrix, CSRMatrix, CSCMatrix)):
            dense_a = self.to_dense()
            dense_b = other.to_dense()

            rows, cols = self.shape
            result = [[0.0] * cols for _ in range(rows)]

            for i in range(rows):
                for j in range(cols):
                    result[i][j] = dense_a[i][j] + dense_b[i][j]

            return COOMatrix.from_dense(result)
        else:
            raise TypeError("Неподдерживаемый тип матрицы для сложения")

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        if scalar == 0:
            return COOMatrix([], [], [], self.shape)

        new_data = [val * scalar for val in self.data]
        return COOMatrix(new_data, self.rows.copy(), self.cols.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        new_shape = (self.shape[1], self.shape[0])
        return COOMatrix(self.data.copy(), self.cols.copy(), self.rows.copy(), new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""
        from CSR import CSRMatrix

        if not isinstance(other, CSRMatrix):
            other_csr = CSRMatrix.from_dense(other.to_dense())
        else:
            other_csr = other

        self_csr = CSRMatrix.from_dense(self.to_dense())

        result_csr = self_csr._matmul_impl(other_csr)

        return COOMatrix.from_dense(result_csr.to_dense())

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

        sorted_indices = sorted(range(self.nnz), key=lambda i: (self.cols[i], self.rows[i]))

        data = []
        indices = []
        indptr = [0] * (self.shape[1] + 1)

        current_col = 0
        for idx in sorted_indices:
            col = self.cols[idx]

            while current_col < col:
                indptr[current_col + 1] = len(data)
                current_col += 1

            data.append(self.data[idx])
            indices.append(self.rows[idx])

        while current_col < self.shape[1]:
            indptr[current_col + 1] = len(data)
            current_col += 1

        return CSCMatrix(data, indices, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix

        sorted_indices = sorted(range(self.nnz), key=lambda i: (self.rows[i], self.cols[i]))

        data = []
        indices = []
        indptr = [0] * (self.shape[0] + 1)

        current_row = 0
        for idx in sorted_indices:
            row = self.rows[idx]

            while current_row < row:
                indptr[current_row + 1] = len(data)
                current_row += 1

            data.append(self.data[idx])
            indices.append(self.cols[idx])

        while current_row < self.shape[0]:
            indptr[current_row + 1] = len(data)
            current_row += 1


        return CSRMatrix(data, indices, indptr, self.shape)
