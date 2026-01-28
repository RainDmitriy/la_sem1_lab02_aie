from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix

class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        
        if len(indptr) != shape[0] + 1:
            raise ValueError(f"indptr должен иметь длину {shape[0] + 1}")
        
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]
        
        for i in range(rows):
            start, end = self.indptr[i], self.indptr[i + 1]
            for idx in range(start, end):
                j = self.indices[idx]
                dense[i][j] = self.data[idx]
        
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        coo_self = self._to_coo()
        coo_other = other._to_coo()
        result_coo = coo_self._add_impl(coo_other)
        return result_coo._to_csr()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        if abs(scalar) < 1e-12:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)
        
        new_data = [val * scalar for val in self.data]
        return CSRMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        from CSC import CSCMatrix
        
        # Простой способ через COO
        coo = self._to_coo()
        transposed_coo = coo.transpose()
        return transposed_coo._to_csc()

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        rows_A, cols_A = self.shape
        rows_B, cols_B = other.shape
        
        result_data, result_indices, result_indptr = [], [], [0] * (rows_A + 1)
        
        for i in range(rows_A):
            row_sum = {}
            
            a_start, a_end = self.indptr[i], self.indptr[i + 1]
            for a_idx in range(a_start, a_end):
                k = self.indices[a_idx]
                a_val = self.data[a_idx]
                
                b_start, b_end = other.indptr[k], other.indptr[k + 1]
                for b_idx in range(b_start, b_end):
                    j = other.indices[b_idx]
                    b_val = other.data[b_idx]
                    
                    row_sum[j] = row_sum.get(j, 0.0) + a_val * b_val
            
            sorted_cols = sorted(row_sum.keys())
            for j in sorted_cols:
                val = row_sum[j]
                if abs(val) > 1e-14:
                    result_data.append(val)
                    result_indices.append(j)
            
            result_indptr[i + 1] = len(result_data)
        
        return CSRMatrix(result_data, result_indices, result_indptr, (rows_A, cols_B))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        rows = len(dense_matrix)
        cols = len(dense_matrix[0])
        
        data, indices = [], []
        row_counts = [0] * rows
        
        for i in range(rows):
            for j in range(cols):
                val = dense_matrix[i][j]
                if abs(val) > 1e-12:
                    data.append(val)
                    indices.append(j)
                    row_counts[i] += 1
        
        indptr = [0] * (rows + 1)
        for i in range(rows):
            indptr[i + 1] = indptr[i] + row_counts[i]
        
        return cls(data, indices, indptr, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        from CSC import CSCMatrix
        
        m, n = self.shape
        col_counts = [0] * n
        
        for col_idx in self.indices:
            col_counts[col_idx] += 1
        
        indptr = [0] * (n + 1)
        for j in range(n):
            indptr[j + 1] = indptr[j] + col_counts[j]
        
        data = [0.0] * len(self.data)
        indices = [0] * len(self.indices)
        current_pos = indptr.copy()
        
        for i in range(m):
            start, end = self.indptr[i], self.indptr[i + 1]
            for k in range(start, end):
                j = self.indices[k]
                val = self.data[k]
                pos = current_pos[j]
                data[pos] = val
                indices[pos] = i
                current_pos[j] += 1
        
        return CSCMatrix(data, indices, indptr, (m, n))

    def _to_coo(self) -> 'COOMatrix':
        from COO import COOMatrix
        
        rows, cols = self.shape
        data, row_indices, col_indices = [], [], []
        
        for i in range(rows):
            start, end = self.indptr[i], self.indptr[i + 1]
            for idx in range(start, end):
                j = self.indices[idx]
                data.append(self.data[idx])
                row_indices.append(i)
                col_indices.append(j)
        
        return COOMatrix(data, row_indices, col_indices, self.shape)