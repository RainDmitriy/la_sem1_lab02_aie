from base import Matrix
from matrix_types import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.nnz = len(data)

    def to_dense(self) -> DenseMatrix:
        mat = [[0.0] * self.shape[1] for _ in range(self.shape[0])]
        for i in range(self.shape[0]):
            for k in range(self.indptr[i], self.indptr[i+1]):
                j = self.indices[k]
                mat[i][j] = self.data[k]
        return mat

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        from collections import defaultdict
        merged = defaultdict(float)
        
        for i in range(self.shape[0]):
            for k in range(self.indptr[i], self.indptr[i+1]):
                merged[(i, self.indices[k])] += self.data[k]
                
        if isinstance(other, CSRMatrix):
             for i in range(other.shape[0]):
                for k in range(other.indptr[i], other.indptr[i+1]):
                    merged[(i, other.indices[k])] += other.data[k]
        else:
            coo = other._to_coo() if hasattr(other, '_to_coo') else other
            for r, c, v in zip(coo.row, coo.col, coo.data):
                merged[(r, c)] += v

        new_data = []
        new_indices = []
        new_indptr = [0]
        
        for i in range(self.shape[0]):
            row_vals = []
            for j in range(self.shape[1]):
                if (i, j) in merged:
                    val = merged[(i, j)]
                    # FIX: Не фильтруем нули
                    row_vals.append((j, val))
            
            row_vals.sort(key=lambda x: x[0])
            for c, v in row_vals:
                new_data.append(v)
                new_indices.append(c)
            new_indptr.append(len(new_data))
            
        return CSRMatrix(new_data, new_indices, new_indptr, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        return CSRMatrix([x * scalar for x in self.data], list(self.indices), list(self.indptr), self.shape)

    def transpose(self) -> 'Matrix':
        from CSC import CSCMatrix
        return CSCMatrix(list(self.data), list(self.indices), list(self.indptr), (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        B = other if isinstance(other, CSRMatrix) else other._to_csr()
        A = self
        
        c_data = []
        c_indices = []
        c_indptr = [0]
        
        for i in range(A.shape[0]):
            row_accum = {} 
            
            for k_a in range(A.indptr[i], A.indptr[i+1]):
                col_a = A.indices[k_a]
                val_a = A.data[k_a]
                
                start_b = B.indptr[col_a]
                end_b = B.indptr[col_a+1]
                
                for k_b in range(start_b, end_b):
                    col_b = B.indices[k_b]
                    val_b = B.data[k_b]
                    row_accum[col_b] = row_accum.get(col_b, 0.0) + val_a * val_b
            
            sorted_cols = sorted(row_accum.keys())
            nnz_row = 0
            for col in sorted_cols:
                val = row_accum[col]
                # FIX: Не фильтруем нули
                c_data.append(val)
                c_indices.append(col)
                nnz_row += 1
            
            c_indptr.append(c_indptr[-1] + nnz_row)
            
        return CSRMatrix(c_data, c_indices, c_indptr, (A.shape[0], B.shape[1]))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        if not dense_matrix:
            return cls([], [], [0], (0, 0))
        rows, cols = len(dense_matrix), len(dense_matrix[0])
        data = []
        indices = []
        indptr = [0]
        for i in range(rows):
            count = 0
            for j in range(cols):
                val = dense_matrix[i][j]
                if val != 0:
                    data.append(val)
                    indices.append(j)
                    count += 1
            indptr.append(indptr[-1] + count)
        return cls(data, indices, indptr, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        from CSC import CSCMatrix
        col_counts = [0] * self.shape[1]
        for idx in self.indices:
            col_counts[idx] += 1
        csc_indptr = [0] * (self.shape[1] + 1)
        cum_sum = 0
        for i in range(self.shape[1]):
            csc_indptr[i] = cum_sum
            cum_sum += col_counts[i]
        csc_indptr[self.shape[1]] = cum_sum
        
        col_cursors = list(csc_indptr)
        csc_data = [0.0] * self.nnz
        csc_indices = [0] * self.nnz
        
        for i in range(self.shape[0]):
            for k in range(self.indptr[i], self.indptr[i+1]):
                col = self.indices[k]
                val = self.data[k]
                dest = col_cursors[col]
                csc_data[dest] = val
                csc_indices[dest] = i 
                col_cursors[col] += 1
                
        return CSCMatrix(csc_data, csc_indices, csc_indptr, self.shape)
    
    def _to_coo(self) -> 'COOMatrix':
        from COO import COOMatrix
        coo_cols = list(self.indices)
        coo_rows = []
        for i in range(self.shape[0]):
            count = self.indptr[i+1] - self.indptr[i]
            coo_rows.extend([i] * count)
        return COOMatrix(list(self.data), coo_rows, coo_cols, self.shape)