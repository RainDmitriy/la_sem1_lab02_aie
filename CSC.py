from base import Matrix
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.nnz = len(data)

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу."""
        n_rows, n_cols = self.shape
        dense = [[0.0] * n_cols for _ in range(n_rows)]
        for j in range(n_cols):
            for k in range(self.indptr[j], self.indptr[j + 1]):
                i = self.indices[k]
                val = self.data[k]
                dense[i][j] = val
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        coo_self = self._to_coo()
        if isinstance(other, CSCMatrix):
            coo_other = other._to_coo()
        else:
            coo_other = other._to_coo()
        return coo_self._add_impl(coo_other)._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        new_data = [val * scalar for val in self.data]
        return CSCMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Hint:
        Результат - в CSR формате (с теми же данными, но с интерпретацией строк как столбцов).
        """
        from CSR import CSRMatrix
        n_rows, n_cols = self.shape
        if self.nnz == 0:
            return CSRMatrix([], [], [0] * (n_cols + 1), (n_cols, n_rows))
        row_counts = [0] * n_cols
        for j in range(n_cols):
            row_counts[j] = self.indptr[j + 1] - self.indptr[j]
        csr_indptr = [0] * (n_cols + 1)
        for i in range(n_cols):
            csr_indptr[i + 1] = csr_indptr[i] + row_counts[i]
        csr_data = [0.0] * self.nnz
        csr_indices = [0] * self.nnz
        current_pos = csr_indptr.copy()
        for j in range(n_cols):
            for k in range(self.indptr[j], self.indptr[j + 1]):
                i = self.indices[k]
                val = self.data[k]
                pos = current_pos[i]
                csr_data[pos] = val
                csr_indices[pos] = j
                current_pos[i] += 1
        return CSRMatrix(csr_data, csr_indices, csr_indptr, (n_cols, n_rows))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        from COO import COOMatrix
        n_rows, n_cols = self.shape
        if isinstance(other, CSCMatrix):
            other_n_rows, other_n_cols = other.shape
            result_n_cols = other_n_cols
            result_data = []
            result_row = []
            result_col = []
            for j in range(result_n_cols):
                column_dict = {}
                for k in range(other.indptr[j], other.indptr[j + 1]):
                    other_row = other.indices[k]
                    other_val = other.data[k]
                    for p in range(self.indptr[other_row], self.indptr[other_row + 1]):
                        self_row = self.indices[p]
                        self_val = self.data[p]

                        if self_row in column_dict:
                            column_dict[self_row] += self_val * other_val
                        else:
                            column_dict[self_row] = self_val * other_val
                for row_idx, val in column_dict.items():
                    if abs(val) > 1e-12:
                        result_data.append(val)
                        result_row.append(row_idx)
                        result_col.append(j)
            result_coo = COOMatrix(result_data, result_row, result_col, (n_rows, result_n_cols))
            return result_coo._to_csc()
        else:
            if not isinstance(other, CSCMatrix):
                other_csc = other._to_csc()
            else:
                other_csc = other
            return self._matmul_impl(other_csc)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        n_rows = len(dense_matrix)
        n_cols = len(dense_matrix[0]) if n_rows > 0 else 0
        data = []
        indices = []
        indptr = [0] * (n_cols + 1)
        col_counts = [0] * n_cols
        for j in range(n_cols):
            for i in range(n_rows):
                if abs(dense_matrix[i][j]) > 1e-12:
                    col_counts[j] += 1
        for j in range(n_cols):
            indptr[j + 1] = indptr[j] + col_counts[j]
        col_positions = indptr.copy()
        for j in range(n_cols):
            for i in range(n_rows):
                val = dense_matrix[i][j]
                if abs(val) > 1e-12:
                    pos = col_positions[j]
                    data.append(val)
                    indices.append(i)
                    col_positions[j] += 1
        return cls(data, indices, indptr, (n_rows, n_cols))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix
        coo = self._to_coo()
        return coo._to_csr()

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        from COO import COOMatrix
        n_rows, n_cols = self.shape
        data = []
        row = []
        col = []
        for j in range(n_cols):
            for k in range(self.indptr[j], self.indptr[j + 1]):
                data.append(self.data[k])
                row.append(self.indices[k])
                col.append(j)
        return COOMatrix(data, row, col, self.shape)
