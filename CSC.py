from base import Matrix
from matrix_types import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.nnz = len(data)

    def to_dense(self) -> DenseMatrix:
        mat = [[0.0] * self.shape[1] for _ in range(self.shape[0])]
        for j in range(self.shape[1]):
            for k in range(self.indptr[j], self.indptr[j+1]):
                i = self.indices[k]
                val = self.data[k]
                mat[i][j] = val
        return mat

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        from collections import defaultdict
        merged = defaultdict(float)
        
        for j in range(self.shape[1]):
            for k in range(self.indptr[j], self.indptr[j+1]):
                merged[(self.indices[k], j)] += self.data[k]
                
        coo_other = other._to_coo() if hasattr(other, '_to_coo') else other
        for r, c, v in zip(coo_other.row, coo_other.col, coo_other.data):
            merged[(r, c)] += v
            
        new_data = []
        new_indices = []
        new_indptr = [0]
        current_ptr = 0
        
        for j in range(self.shape[1]):
            col_vals = []
            for i in range(self.shape[0]):
                if (i, j) in merged:
                    val = merged[(i, j)]
                    col_vals.append((i, val))
            
            col_vals.sort(key=lambda x: x[0])
            for r, v in col_vals:
                new_data.append(v)
                new_indices.append(r)
                current_ptr += 1
            new_indptr.append(current_ptr)
            
        return CSCMatrix(new_data, new_indices, new_indptr, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        return CSCMatrix([x * scalar for x in self.data], list(self.indices), list(self.indptr), self.shape)

    def transpose(self) -> 'Matrix':
        from CSR import CSRMatrix
        return CSRMatrix(list(self.data), list(self.indices), list(self.indptr), (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        B = other if isinstance(other, CSCMatrix) else other._to_csc()
        A = self
        
        c_data = []
        c_indices = []
        c_indptr = [0]
        
        for j in range(B.shape[1]):
            col_accum = {}
            
            for k_b in range(B.indptr[j], B.indptr[j+1]):
                row_b = B.indices[k_b] 
                val_b = B.data[k_b]
                
                start_a = A.indptr[row_b]
                end_a = A.indptr[row_b+1]
                
                for k_a in range(start_a, end_a):
                    row_a = A.indices[k_a]
                    val_a = A.data[k_a]
                    col_accum[row_a] = col_accum.get(row_a, 0.0) + val_a * val_b
            
            sorted_rows = sorted(col_accum.keys())
            nnz_in_col = 0
            for r in sorted_rows:
                val = col_accum[r]
                # FIX: Не фильтруем нули
                c_data.append(val)
                c_indices.append(r)
                nnz_in_col += 1
            
            c_indptr.append(c_indptr[-1] + nnz_in_col)
            
        return CSCMatrix(c_data, c_indices, c_indptr, (A.shape[0], B.shape[1]))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        if not dense_matrix:
            return cls([], [], [0], (0, 0))
        rows, cols = len(dense_matrix), len(dense_matrix[0])
        data = []
        indices = []
        indptr = [0]
        count = 0
        for j in range(cols):
            for i in range(rows):
                val = dense_matrix[i][j]
                if val != 0:
                    data.append(val)
                    indices.append(i)
                    count += 1
            indptr.append(count)
        return cls(data, indices, indptr, (rows, cols))

    def _to_csr(self) -> 'CSRMatrix':
        from CSR import CSRMatrix
        row_counts = [0] * self.shape[0]
        for idx in self.indices:
            row_counts[idx] += 1
        csr_indptr = [0] * (self.shape[0] + 1)
        cum_sum = 0
        for i in range(self.shape[0]):
            csr_indptr[i] = cum_sum
            cum_sum += row_counts[i]
        csr_indptr[self.shape[0]] = cum_sum
        
        row_cursors = list(csr_indptr)
        csr_data = [0.0] * self.nnz
        csr_indices = [0] * self.nnz
        
        for j in range(self.shape[1]):
            for k in range(self.indptr[j], self.indptr[j+1]):
                row = self.indices[k]
                val = self.data[k]
                dest = row_cursors[row]
                csr_data[dest] = val
                csr_indices[dest] = j 
                row_cursors[row] += 1
                
        return CSRMatrix(csr_data, csr_indices, csr_indptr, self.shape)

    def _to_coo(self) -> 'COOMatrix':
        from COO import COOMatrix
        coo_rows = list(self.indices)
        coo_cols = []
        for j in range(self.shape[1]):
            count = self.indptr[j+1] - self.indptr[j]
            coo_cols.extend([j] * count)
        return COOMatrix(list(self.data), coo_rows, coo_cols, self.shape)