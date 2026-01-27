from base import Matrix
from types import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        rows, cols = shape
        if len(indptr) != rows + 1:
            raise ValueError("indptr has wrong length")

        if len(data) != len(indices):
            raise ValueError("data and indices dont have same length")

        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        rows, cols = self.shape
        matrix = []
        for i in range(rows):
            row = []
            for j in range(cols):
                row.append(0.0)
            matrix.append(row)

        for row in range(rows):
            start = self.indptr[row]
            end = self.indptr[row + 1]
            for idx in range(start, end):
                value = self.data[idx]
                col_index = self.indices[idx]
                matrix[row][col_index] = value
        return matrix

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        dense_self = self.to_dense()
        dense_other = other.to_dense()
        rows, cols = self.shape
        result_dense = []
        for i in range(rows):
            new_row = []
            for j in range(cols):
                sum_val = dense_self[i][j] + dense_other[i][j]
                new_row.append(sum_val)
            result_dense.append(new_row)

        return CSRMatrix.from_dense(result_dense)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        if scalar == 0.0:
            rows, cols = self.shape
            return CSRMatrix([], [], [0] * (rows + 1), self.shape)
        new_data = [x * scalar for x in self.data]
        return CSRMatrix(new_data, self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Hint:
        Результат - в CSC формате (с теми же данными, но с интерпретацией столбцов как строк).
        """
        from CSC import CSCMatrix
        new_shape = (self.shape[1], self.shape[0])
        return CSCMatrix(
            data=self.data[:],
            indices=self.indices[:],
            indptr=self.indptr[:],
            shape=new_shape
        )

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        A = self.to_dense()
        B = other.to_dense()
        rows_A, cols_A = self.shape
        rows_B, cols_B = other.shape
        if cols_A != rows_B:
            raise ValueError("matrix A' col and matrix B' row doesnt have same length")

        result = []
        for i in range(rows_A):
            new_row = []
            for j in range(cols_B):
                total = 0.0
                for k in range(cols_A):
                    total += A[i][k] * B[k][j]
                new_row.append(total)
            result.append(new_row)
        return CSRMatrix.from_dense(result)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        if not dense_matrix:
            return cls([], [], [0], (0, 0))
        rows = len(dense_matrix)
        if rows == 0:
            cols = 0
        else:
            cols = len(dense_matrix[0])
        data, indices, indptr = [], [], [0]

        for row in range(rows):
            for col in range(cols):
                value = dense_matrix[row][col]
                if value != 0.0:
                    data.append(value)
                    indices.append(col)
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
        from COO import COOMatrix
        rows, cols = self.shape
        data = []
        row_indices = []
        col_indices = []

        for i in range(rows):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for idx in range(start, end):
                data.append(self.data[idx])
                row_indices.append(i)
                col_indices.append(self.indices[idx])

        return COOMatrix(data, row_indices, col_indices, (rows, cols))