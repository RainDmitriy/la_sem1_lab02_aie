from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.nnz = len(data)

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        n_rows, n_cols = self.shape
        dense = [[0.0] * n_cols for _ in range(n_rows)]
        for i in range(n_rows):
            for k in range(self.indptr[i], self.indptr[i + 1]):
                j = self.indices[k]
                val = self.data[k]
                dense[i][j] = val
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        data_self = list(self.data)
        indices_self = list(self.indices)
        indptr_self = list(self.indptr)
        data_other = list(other.data)
        indices_other = list(other.indices)
        indptr_other = list(other.indptr)
        result_data = []
        result_indices = []
        result_indptr = [0]
        rows, cols = self.shape
        for r in range(rows):
            self_start = indptr_self[r]
            other_start = indptr_other[r]
            self_end = indptr_self[r + 1]
            other_end = indptr_other[r + 1]
            while self_start < self_end or other_start < other_end:
                if self_start == self_end:
                    c, val = indices_other[other_start], data_other[other_start]
                    other_start += 1
                elif other_start == other_end:
                    c, val = indices_self[self_start], data_self[self_start]
                    self_start += 1
                elif indices_self[self_start] < indices_other[other_start]:
                    c, val = indices_self[self_start], data_self[self_start]
                    self_start += 1
                elif indices_other[other_start] < indices_self[self_start]:
                    c, val = indices_other[other_start], data_other[other_start]
                    other_start += 1
                else:
                    c, val = indices_self[self_start], data_self[self_start] + data_other[other_start]
                    self_start += 1
                    other_start += 1
                result_indices.append(c)
                result_data.append(val)
            result_indptr.append(len(result_data))
        return CSRMatrix(result_data, result_indices, result_indptr, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        new_data = [val * scalar for val in self.data]
        return CSRMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Hint:
        Результат - в CSC формате (с теми же данными, но с интерпретацией столбцов как строк).
        """
        n_rows, n_cols = self.shape
        from COO import COOMatrix
        coo_data = []
        coo_row = []
        coo_col = []
        for i in range(n_rows):
            for k in range(self.indptr[i], self.indptr[i + 1]):
                coo_data.append(self.data[k])
                coo_row.append(i)
                coo_col.append(self.indices[k])
        coo = COOMatrix(coo_data, coo_row, coo_col, (n_rows, n_cols))
        coo_t = coo.transpose()
        return coo_t._to_csc()

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        from COO import COOMatrix
        n_rows, n_cols = self.shape
        if isinstance(other, CSRMatrix):
            other_n_rows, other_n_cols = other.shape
            result_n_cols = other_n_cols
            result_data = []
            result_row = []
            result_col = []
            for i in range(n_rows):
                row_dict = {}
                for k in range(self.indptr[i], self.indptr[i + 1]):
                    self_col = self.indices[k]
                    self_val = self.data[k]
                    for p in range(other.indptr[self_col], other.indptr[self_col + 1]):
                        other_col = other.indices[p]
                        other_val = other.data[p]
                        if other_col in row_dict:
                            row_dict[other_col] += self_val * other_val
                        else:
                            row_dict[other_col] = self_val * other_val
                for col_idx, val in row_dict.items():
                    if abs(val) > 1e-12:
                        result_data.append(val)
                        result_row.append(i)
                        result_col.append(col_idx)
            result_coo = COOMatrix(result_data, result_row, result_col, (n_rows, result_n_cols))
            return result_coo._to_csr()
        else:
            if not isinstance(other, CSRMatrix):
                other_csr = other._to_csr()
            else:
                other_csr = other
            return self._matmul_impl(other_csr)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        n_rows = len(dense_matrix)
        n_cols = len(dense_matrix[0]) if n_rows > 0 else 0
        data = []
        indices = []
        indptr = [0] * (n_rows + 1)
        row_counts = [0] * n_rows
        for i in range(n_rows):
            for j in range(n_cols):
                if abs(dense_matrix[i][j]) > 1e-12:
                    row_counts[i] += 1
        for i in range(n_rows):
            indptr[i + 1] = indptr[i] + row_counts[i]
        row_positions = indptr.copy()
        for i in range(n_rows):
            for j in range(n_cols):
                val = dense_matrix[i][j]
                if abs(val) > 1e-12:
                    pos = row_positions[i]
                    data.append(val)
                    indices.append(j)
                    row_positions[i] += 1
        return cls(data, indices, indptr, (n_rows, n_cols))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSRMatrix в CSCMatrix.
        """
        coo = self._to_coo()
        return coo._to_csc()

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSRMatrix в COOMatrix.
        """
        from COO import COOMatrix
        n_rows, n_cols = self.shape
        data = []
        row = []
        col = []
        for i in range(n_rows):
            for k in range(self.indptr[i], self.indptr[i + 1]):
                data.append(self.data[k])
                row.append(i)
                col.append(self.indices[k])
        return COOMatrix(data, row, col, self.shape)
