from base import Matrix
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix

class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        
        if len(indptr) != shape[1] + 1:
            raise ValueError(f"indptr должен иметь длину {shape[1] + 1}")
        
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]
        
        for j in range(cols):
            start, end = self.indptr[j], self.indptr[j + 1]
            for idx in range(start, end):
                i = self.indices[idx]
                dense[i][j] = self.data[idx]
        
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        coo_self = self._to_coo()
        coo_other = other._to_coo()
        result_coo = coo_self._add_impl(coo_other)
        return result_coo._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        if abs(scalar) < 1e-12:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)
        
        new_data = [val * scalar for val in self.data]
        return CSCMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        from CSR import CSRMatrix
        
        # Простой способ через COO
        coo = self._to_coo()
        transposed_coo = coo.transpose()
        return transposed_coo._to_csr()

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        rows_A, cols_A = self.shape
        rows_B, cols_B = other.shape
        
        # Конвертируем в CSR для умножения
        csr_self = self._to_csr()
        if isinstance(other, CSCMatrix):
            csr_other = other._to_csr()
        else:
            csr_other = other._to_csr()
        
        result_csr = csr_self._matmul_impl(csr_other)
        return result_csr._to_csc()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        rows = len(dense_matrix)
        cols = len(dense_matrix[0])
        
        data, indices = [], []
        col_counts = [0] * cols
        
        for j in range(cols):
            for i in range(rows):
                val = dense_matrix[i][j]
                if abs(val) > 1e-12:
                    data.append(val)
                    indices.append(i)
                    col_counts[j] += 1
        
        indptr = [0] * (cols + 1)
        for j in range(cols):
            indptr[j + 1] = indptr[j] + col_counts[j]
        
        return cls(data, indices, indptr, (rows, cols))

    def _to_csr(self) -> 'CSRMatrix':
        from CSR import CSRMatrix
        
        m, n = self.shape
        row_counts = [0] * m
        
        for row_idx in self.indices:
            row_counts[row_idx] += 1
        
        indptr = [0] * (m + 1)
        for i in range(m):
            indptr[i + 1] = indptr[i] + row_counts[i]
        
        data = [0.0] * len(self.data)
        indices = [0] * len(self.indices)
        current_pos = indptr.copy()
        
        for j in range(n):
            start, end = self.indptr[j], self.indptr[j + 1]
            for k in range(start, end):
                i = self.indices[k]
                val = self.data[k]
                pos = current_pos[i]
                data[pos] = val
                indices[pos] = j
                current_pos[i] += 1
        
        return CSRMatrix(data, indices, indptr, (m, n))

    def _to_coo(self) -> 'COOMatrix':
        from COO import COOMatrix
        
        rows, cols = self.shape
        data, row_indices, col_indices = [], [], []
        
        for j in range(cols):
            start, end = self.indptr[j], self.indptr[j + 1]
            for idx in range(start, end):
                i = self.indices[idx]
                data.append(self.data[idx])
                row_indices.append(i)
                col_indices.append(j)
        
        return COOMatrix(data, row_indices, col_indices, self.shape)