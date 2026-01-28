from base import Matrix
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from COO import COOMatrix
    from CSR import CSRMatrix

class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        if len(indptr) != shape[1] + 1:
            raise ValueError()
        if indptr[0] != 0:
            raise ValueError()
        if indptr[-1] != len(data):
            raise ValueError()
        if len(data) != len(indices):
            raise ValueError()
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0] * cols for _ in range(rows)]
        for j in range(cols):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for idx in range(start, end):
                i = self.indices[idx]
                value = self.data[idx]
                dense[i][j] = value
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        self_coo = self._to_coo()
        other_coo = other._to_coo()
        result_coo = self_coo._add_impl(other_coo)
        return result_coo._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        if scalar == 0:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)
        new_data = [value * scalar for value in self.data]
        return CSCMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Hint:
        Результат - в CSR формате (с теми же данными, но с интерпретацией строк как столбцов).
        """
        from CSR import CSRMatrix
        rows, cols = self.shape
        new_rows, new_cols = cols, rows
        row_counts = [0] * new_rows
        for j in range(cols):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            row_counts[j] = end - start
        new_indptr = [0] * (new_rows + 1)
        for i in range(new_rows):
            new_indptr[i + 1] = new_indptr[i] + row_counts[i]
        new_data = [0] * len(self.data)
        new_indices = [0] * len(self.indices)
        row_positions = new_indptr.copy()
        for j in range(cols):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for idx in range(start, end):
                i = self.indices[idx]
                value = self.data[idx]
                pos = row_positions[j]
                new_data[pos] = value
                new_indices[pos] = i
                row_positions[j] += 1
        return CSRMatrix(new_data, new_indices, new_indptr, (new_rows, new_cols))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        rows_A, cols_A = self.shape
        rows_B, cols_B = other.shape
        result_data = []
        result_indices = []
        result_indptr = [0] * (cols_B + 1)
        row_entries_B = [[] for _ in range(rows_B)]
        for col in range(cols_B):
            start = other.indptr[col]
            end = other.indptr[col + 1]
            for idx in range(start, end):
                row = other.indices[idx]
                value = other.data[idx]
                row_entries_B[row].append((col, value))
        temp_row = [0.0] * rows_A
        for j in range(cols_B):
            for i in range(rows_A):
                temp_row[i] = 0.0
            for i in range(rows_B):
                for col_b, val_b in row_entries_B[i]:
                    if col_b == j:
                        col_start = self.indptr[i]
                        col_end = self.indptr[i + 1]
                        for a_idx in range(col_start, col_end):
                            row_a = self.indices[a_idx]
                            val_a = self.data[a_idx]
                            temp_row[row_a] += val_a * val_b
            for i in range(rows_A):
                if abs(temp_row[i]) > 1e-14:
                    result_data.append(temp_row[i])
                    result_indices.append(i)
            result_indptr[j + 1] = len(result_data)
        return CSCMatrix(result_data, result_indices, result_indptr, (rows_A, cols_B))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0])
        data = []
        indices = []
        col_counts = [0] * cols
        for j in range(cols):
            for i in range(rows):
                value = dense_matrix[i][j]
                if value != 0:
                    data.append(value)
                    indices.append(i)
                    col_counts[j] += 1
        indptr = [0] * (cols + 1)
        for j in range(cols):
            indptr[j + 1] = indptr[j] + col_counts[j]
        return CSCMatrix(data, indices, indptr, (rows, cols))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix
        m, n = self.shape
        row_counts = [0] * m
        for row_idx in self.indices:
            row_counts[row_idx] += 1
        indptr = [0] * (m + 1)
        for i in range(m):
            indptr[i + 1] = indptr[i] + row_counts[i]
        data = [0] * len(self.data)
        indices = [0] * len(self.indices)
        current_pos = indptr.copy()
        for j in range(n):
            col_start = self.indptr[j]
            col_end = self.indptr[j + 1]
            for k in range(col_start, col_end):
                i = self.indices[k]
                val = self.data[k]
                pos = current_pos[i]
                data[pos] = val
                indices[pos] = j
                current_pos[i] += 1
        return CSRMatrix(data, indices, indptr, (m, n))

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        from COO import COOMatrix
        rows, cols = self.shape
        data_list = []
        row_indices = []
        col_indices = []
        for j in range(cols):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for idx in range(start, end):
                i = self.indices[idx]
                value = self.data[idx]
                data_list.append(value)
                row_indices.append(i)
                col_indices.append(j)
        return COOMatrix(data_list, row_indices, col_indices, self.shape)