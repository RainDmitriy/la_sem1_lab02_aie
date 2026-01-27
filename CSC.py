from base import Matrix
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data.copy()
        self.indices = indices.copy()
        self.indptr = indptr.copy()

    def to_dense(self) -> DenseMatrix:
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]

        for j in range(cols):
            for idx in range(self.indptr[j], self.indptr[j + 1]):
                i = self.indices[idx]
                dense[i][j] = self.data[idx]

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        csr_self = self._to_csr()
        if isinstance(other, CSCMatrix):
            csr_other = other._to_csr()
        else:
            from CSR import CSRMatrix
            csr_other = CSRMatrix.from_dense(other.to_dense())

        result_csr = csr_self._add_impl(csr_other)
        return result_csr._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        if scalar == 0:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)
        new_data = [val * scalar for val in self.data]
        return CSCMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        from CSR import CSRMatrix

        if len(self.data) == 0:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), (self.shape[1], self.shape[0]))

        rows, cols = self.shape
        nnz = len(self.data)

        data_csr = [0.0] * nnz
        indices_csr = [0] * nnz
        indptr_csr = [0] * (rows + 1)

        for i in self.indices:
            indptr_csr[i + 1] += 1

        for i in range(1, rows + 1):
            indptr_csr[i] += indptr_csr[i - 1]

        current_pos = indptr_csr.copy()
        for j in range(cols):
            for idx in range(self.indptr[j], self.indptr[j + 1]):
                i = self.indices[idx]
                pos = current_pos[i]
                data_csr[pos] = self.data[idx]
                indices_csr[pos] = j
                current_pos[i] += 1

        return CSRMatrix(data_csr, indices_csr, indptr_csr, (cols, rows))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        csr_self = self._to_csr()
        if isinstance(other, CSCMatrix):
            csr_other = other._to_csr()
        else:
            from CSR import CSRMatrix
            csr_other = CSRMatrix.from_dense(other.to_dense())

        result_csr = csr_self._matmul_impl(csr_other)
        return result_csr._to_csc()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        data = []
        indices = []
        indptr = [0]

        for j in range(cols):
            for i in range(rows):
                val = dense_matrix[i][j]
                if val != 0:
                    data.append(val)
                    indices.append(i)
            indptr.append(len(data))

        return cls(data, indices, indptr, (rows, cols))

    def _to_csr(self) -> 'CSRMatrix':
        return self.transpose()

    def _to_coo(self) -> 'COOMatrix':
        from COO import COOMatrix

        rows, cols = self.shape
        data = []
        row_indices = []
        col_indices = []

        for j in range(cols):
            for idx in range(self.indptr[j], self.indptr[j + 1]):
                data.append(self.data[idx])
                row_indices.append(self.indices[idx])
                col_indices.append(j)

        return COOMatrix(data, row_indices, col_indices, self.shape)