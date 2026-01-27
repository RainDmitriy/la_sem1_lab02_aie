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
        from COO import COOMatrix
        coo_self = self._to_coo()

        if isinstance(other, CSRMatrix):
            coo_other = other._to_coo()
        else:
            coo_other = COOMatrix.from_dense(other.to_dense())

        result_coo = coo_self._add_impl(coo_other)
        return result_coo._to_csr()

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
        from COO import COOMatrix
        coo_self = self._to_coo()

        if isinstance(other, CSRMatrix):
            coo_other = other._to_coo()
        else:
            coo_other = COOMatrix.from_dense(other.to_dense())

        result_coo = coo_self._matmul_impl(coo_other)
        return result_coo._to_csr()

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