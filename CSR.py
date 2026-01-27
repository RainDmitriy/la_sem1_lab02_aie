from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data.copy()
        self.indices = indices.copy()
        self.indptr = indptr.copy()

    def to_dense(self) -> DenseMatrix:
        """Преобразует разреженную матрицу в плотную."""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]
        for i in range(rows):
            for idx in range(self.indptr[i], self.indptr[i + 1]):
                j = self.indices[idx]
                dense[i][j] = self.data[idx]
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Реализация сложения с другой матрицей."""
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
                        if abs(val) > 1e-12:
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
            from COO import COOMatrix
            coo_self = self._to_coo()
            coo_other = COOMatrix.from_dense(other.to_dense())
            return coo_self._add_impl(coo_other)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Реализация умножения на скаляр."""
        if abs(scalar) < 1e-12:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)
        new_data = [val * scalar for val in self.data]
        return CSRMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование матрицы."""
        from CSC import CSCMatrix

        if len(self.data) == 0:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), (self.shape[1], self.shape[0]))

        rows, cols = self.shape
        new_rows, new_cols = cols, rows

        col_counts = [0] * new_rows

        for i in range(rows):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            col_counts[i] = end - start

        new_indptr = [0] * (new_rows + 1)
        for j in range(new_rows):
            new_indptr[j + 1] = new_indptr[j] + col_counts[j]

        new_data = [0.0] * len(self.data)
        new_indices = [0] * len(self.indices)

        col_positions = new_indptr.copy()

        for i in range(rows):
            start = self.indptr[i]
            end = self.indptr[i + 1]

            for idx in range(start, end):
                j = self.indices[idx]
                value = self.data[idx]

                pos = col_positions[i]
                new_data[pos] = value
                new_indices[pos] = j
                col_positions[i] += 1

        return CSCMatrix(new_data, new_indices, new_indptr, (new_rows, new_cols))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Реализация умножения матриц."""
        if isinstance(other, CSRMatrix):
            A_rows, A_cols = self.shape
            B_rows, B_cols = other.shape

            if A_cols != B_rows:
                raise ValueError("Несовместимые размерности для умножения")

            B_T = other.transpose()

            result_data, result_indices = [], []
            result_indptr = [0]

            for i in range(A_rows):
                row_result = {}

                for a_idx in range(self.indptr[i], self.indptr[i + 1]):
                    k = self.indices[a_idx]
                    a_val = self.data[a_idx]

                    if k < B_T.shape[0]:
                        for b_idx in range(B_T.indptr[k], B_T.indptr[k + 1]):
                            j = B_T.indices[b_idx]
                            b_val = B_T.data[b_idx]

                            if j in row_result:
                                row_result[j] += a_val * b_val
                            else:
                                row_result[j] = a_val * b_val

                sorted_cols = sorted(row_result.keys())
                for j in sorted_cols:
                    val = row_result[j]
                    if abs(val) > 1e-12:
                        result_data.append(val)
                        result_indices.append(j)

                result_indptr.append(len(result_data))

            return CSRMatrix(result_data, result_indices, result_indptr, (A_rows, B_cols))
        else:
            from COO import COOMatrix
            from CSC import CSCMatrix

            if isinstance(other, COOMatrix):
                other_csr = other._to_csr()
            elif isinstance(other, CSCMatrix):
                other_csr = other._to_csr()
            else:
                other_csr = CSRMatrix.from_dense(other.to_dense())

            return self._matmul_impl(other_csr)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        data, indices, indptr = [], [], [0]

        for i in range(rows):
            for j in range(cols):
                val = dense_matrix[i][j]
                if abs(val) > 1e-12:
                    data.append(val)
                    indices.append(j)
            indptr.append(len(data))

        return cls(data, indices, indptr, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """Преобразование CSR в CSC."""
        return self.transpose()

    def _to_coo(self) -> 'COOMatrix':
        """Преобразование CSR в COO."""
        from COO import COOMatrix

        rows, cols = self.shape
        data, row_indices, col_indices = [], [], []

        for i in range(rows):
            for idx in range(self.indptr[i], self.indptr[i + 1]):
                data.append(self.data[idx])
                row_indices.append(i)
                col_indices.append(self.indices[idx])

        return COOMatrix(data, row_indices, col_indices, self.shape)