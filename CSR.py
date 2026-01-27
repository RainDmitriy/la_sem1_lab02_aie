from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        '''Преобразует CSR в плотную матрицу.'''
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]
        for i in range(rows):
            for idx in range(self.indptr[i], self.indptr[i + 1]):
                j = self.indices[idx]
                dense[i][j] = self.data[idx]
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        '''Сложение CSR матриц.'''
        from COO import COOMatrix
        from CSC import CSCMatrix

        if isinstance(other, CSRMatrix):
            rows, cols = self.shape
            result_data, result_indices = [], []
            result_indptr = [0]

            for i in range(rows):
                idx1, idx2 = self.indptr[i], other.indptr[i]
                end1, end2 = self.indptr[i + 1], other.indptr[i + 1]

                while idx1 < end1 and idx2 < end2:
                    col1, col2 = self.indices[idx1], other.indices[idx2]

                    if col1 < col2:
                        result_data.append(self.data[idx1])
                        result_indices.append(col1)
                        idx1 += 1
                    elif col2 < col1:
                        result_data.append(other.data[idx2])
                        result_indices.append(col2)
                        idx2 += 1
                    else:
                        val = self.data[idx1] + other.data[idx2]
                        if abs(val) > 1e-14:
                            result_data.append(val)
                            result_indices.append(col1)
                        idx1 += 1
                        idx2 += 1

                while idx1 < end1:
                    result_data.append(self.data[idx1])
                    result_indices.append(self.indices[idx1])
                    idx1 += 1

                while idx2 < end2:
                    result_data.append(other.data[idx2])
                    result_indices.append(other.indices[idx2])
                    idx2 += 1

                result_indptr.append(len(result_data))

            return CSRMatrix(result_data, result_indices, result_indptr, self.shape)
        else:
            coo_self = self._to_coo()
            coo_other = COOMatrix.from_dense(other.to_dense())
            result_coo = coo_self._add_impl(coo_other)
            return result_coo._to_csr()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        '''Умножение CSR на скаляр.'''
        if abs(scalar) < 1e-14:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)
        new_data = [val * scalar for val in self.data]
        return CSRMatrix(new_data, self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        '''
        Транспонирование CSR матрицы.
        Hint:
        Результат - в CSC формате (с теми же данными, но с интерпретацией столбцов как строк).
        '''
        from CSC import CSCMatrix
        return self._to_coo().transpose()._to_csc()

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        '''Умножение CSR матриц.'''
        from COO import COOMatrix
        from CSC import CSCMatrix

        if isinstance(other, CSRMatrix):
            A_rows, A_cols = self.shape
            B_rows, B_cols = other.shape

            result_data, result_indices = [], []
            result_indptr = [0] * (A_rows + 1)

            for i in range(A_rows):
                row_result = {}
                row_start = self.indptr[i]
                row_end = self.indptr[i + 1]

                for a_idx in range(row_start, row_end):
                    k = self.indices[a_idx]
                    a_val = self.data[a_idx]

                    b_row_start = other.indptr[k]
                    b_row_end = other.indptr[k + 1]

                    for b_idx in range(b_row_start, b_row_end):
                        j = other.indices[b_idx]
                        b_val = other.data[b_idx]

                        row_result[j] = row_result.get(j, 0.0) + a_val * b_val

                sorted_cols = sorted(row_result.keys())
                for j in sorted_cols:
                    val = row_result[j]
                    if abs(val) > 1e-14:
                        result_data.append(val)
                        result_indices.append(j)

                result_indptr[i + 1] = len(result_data)

            return CSRMatrix(result_data, result_indices, result_indptr, (A_rows, B_cols))
        else:
            coo_self = self._to_coo()
            coo_other = COOMatrix.from_dense(other.to_dense())
            result_coo = coo_self._matmul_impl(coo_other)
            return result_coo._to_csr()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        '''Создание CSR из плотной матрицы.'''
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        data, indices = [], []
        indptr = [0]

        for i in range(rows):
            for j in range(cols):
                val = dense_matrix[i][j]
                if abs(val) > 1e-14:
                    data.append(val)
                    indices.append(j)
            indptr.append(len(data))

        return cls(data, indices, indptr, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        '''
        Преобразование CSRMatrix в CSCMatrix.
        '''
        from CSC import CSCMatrix

        m, n = self.shape

        col_counts = [0] * n
        for col_idx in self.indices:
            col_counts[col_idx] += 1

        indptr = [0] * (n + 1)
        for j in range(n):
            indptr[j + 1] = indptr[j] + col_counts[j]

        data = [0] * len(self.data)
        indices = [0] * len(self.indices)

        current_pos = indptr[:]

        for i in range(m):
            row_start = self.indptr[i]
            row_end = self.indptr[i + 1]

            for k in range(row_start, row_end):
                j = self.indices[k]
                val = self.data[k]

                pos = current_pos[j]
                data[pos] = val
                indices[pos] = i
                current_pos[j] += 1

        return CSCMatrix(data, indices, indptr, (m, n))

    def _to_coo(self) -> 'COOMatrix':
        '''
        Преобразование CSRMatrix в COOMatrix.
        '''
        from COO import COOMatrix

        rows, cols = self.shape
        data, row_indices, col_indices = [], [], []

        for i in range(rows):
            for idx in range(self.indptr[i], self.indptr[i + 1]):
                data.append(self.data[idx])
                row_indices.append(i)
                col_indices.append(self.indices[idx])

        return COOMatrix(data, row_indices, col_indices, self.shape)