from base import Matrix
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data.copy()
        self.indices = indices.copy()
        self.indptr = indptr.copy()

    def to_dense(self) -> DenseMatrix:
        """Преобразует разреженную матрицу в плотную."""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]
        for j in range(cols):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for idx in range(start, end):
                i = self.indices[idx]
                dense[i][j] = self.data[idx]
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Реализация сложения с другой матрицей."""
        from COO import COOMatrix
        from CSR import CSRMatrix

        if isinstance(other, CSCMatrix):
            csr_self = self._to_csr()
            csr_other = other._to_csr()
            result_csr = csr_self._add_impl(csr_other)
            return result_csr._to_csc()
        else:
            coo_self = self._to_coo()

            if isinstance(other, COOMatrix):
                coo_other = other
            elif isinstance(other, CSRMatrix):
                coo_other = other._to_coo()
            else:
                coo_other = COOMatrix.from_dense(other.to_dense())

            result_coo = coo_self._add_impl(coo_other)
            return result_coo._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Реализация умножения на скаляр."""
        if abs(scalar) < 1e-12:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)
        new_data = [val * scalar for val in self.data]
        return CSCMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование матрицы."""
        from CSR import CSRMatrix

        if len(self.data) == 0:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), (self.shape[1], self.shape[0]))

        rows, cols = self.shape
        row_counts = [0] * rows

        for j in range(cols):
            for idx in range(self.indptr[j], self.indptr[j + 1]):
                row_counts[self.indices[idx]] += 1

        indptr_T = [0] * (rows + 1)
        for i in range(rows):
            indptr_T[i + 1] = indptr_T[i] + row_counts[i]

        data_T = [0.0] * len(self.data)
        indices_T = [0] * len(self.indices)

        current_pos = indptr_T.copy()

        for j in range(cols):
            for idx in range(self.indptr[j], self.indptr[j + 1]):
                i = self.indices[idx]
                pos = current_pos[i]
                data_T[pos] = self.data[idx]
                indices_T[pos] = j
                current_pos[i] += 1

        return CSRMatrix(data_T, indices_T, indptr_T, (cols, rows))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Реализация умножения матриц."""
        from CSR import CSRMatrix
        from COO import COOMatrix

        csr_self = self._to_csr()

        if isinstance(other, CSCMatrix):
            csr_other = other._to_csr()
        elif isinstance(other, COOMatrix):
            csr_other = other._to_csr()
        elif isinstance(other, CSRMatrix):
            csr_other = other
        else:
            csr_other = CSRMatrix.from_dense(other.to_dense())

        result_csr = csr_self._matmul_impl(csr_other)
        return result_csr._to_csc()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        data, indices = [], []
        indptr = [0]

        col_counts = [0] * cols

        for j in range(cols):
            count = 0
            for i in range(rows):
                val = dense_matrix[i][j]
                if abs(val) > 1e-12:
                    data.append(val)
                    indices.append(i)
                    count += 1
            col_counts[j] = count

        for j in range(cols):
            indptr.append(indptr[j] + col_counts[j])

        return cls(data, indices, indptr, (rows, cols))

    def _to_csr(self) -> 'CSRMatrix':
        """Преобразование CSC в CSR."""
        from CSR import CSRMatrix

        rows, cols = self.shape
        row_counts = [0] * rows

        for j in range(cols):
            for idx in range(self.indptr[j], self.indptr[j + 1]):
                row_counts[self.indices[idx]] += 1

        indptr = [0] * (rows + 1)
        for i in range(rows):
            indptr[i + 1] = indptr[i] + row_counts[i]

        data = [0.0] * len(self.data)
        indices = [0] * len(self.indices)

        current_pos = indptr.copy()

        for j in range(cols):
            for idx in range(self.indptr[j], self.indptr[j + 1]):
                i = self.indices[idx]
                pos = current_pos[i]
                data[pos] = self.data[idx]
                indices[pos] = j
                current_pos[i] += 1

        return CSRMatrix(data, indices, indptr, self.shape)

    def _to_coo(self) -> 'COOMatrix':
        """Преобразование CSC в COO."""
        from COO import COOMatrix

        rows, cols = self.shape
        data, row_indices, col_indices = [], [], []

        for j in range(cols):
            for idx in range(self.indptr[j], self.indptr[j + 1]):
                data.append(self.data[idx])
                row_indices.append(self.indices[idx])
                col_indices.append(j)

        return COOMatrix(data, row_indices, col_indices, self.shape)