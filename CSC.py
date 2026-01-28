from typing import List, Tuple
from base import Matrix
from type import DenseMatrix, Shape, Vector, CSCData, CSCIndices, CSCIndptr, EPS


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.n_rows, self.n_cols = shape
        
        if len(indptr) != self.n_cols + 1:
            raise ValueError(f"indptr должен иметь длину {self.n_cols + 1}, получено {len(indptr)}")
        
        if len(data) != len(indices):
            raise ValueError("data и indices должны иметь одинаковую длину")
    
    def to_dense(self) -> DenseMatrix:
        dense = [[0.0] * self.n_cols for _ in range(self.n_rows)]
        
        for j in range(self.n_cols):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for idx in range(start, end):
                i = self.indices[idx]
                dense[i][j] = self.data[idx]
        
        return dense
    
    @classmethod
    def from_dense(cls, matrix: DenseMatrix) -> 'CSCMatrix':
        n_rows = len(matrix)
        n_cols = len(matrix[0]) if n_rows > 0 else 0
        
        # Подсчитываем количество элементов в каждом столбце
        col_counts = [0] * n_cols
        for i in range(n_rows):
            for j in range(n_cols):
                if abs(matrix[i][j]) > EPS:
                    col_counts[j] += 1
        
        # Вычисляем indptr
        indptr = [0] * (n_cols + 1)
        for j in range(n_cols):
            indptr[j + 1] = indptr[j] + col_counts[j]
        
        # Рабочий массив для заполнения
        next_pos = indptr.copy()
        data = [0.0] * sum(col_counts)
        indices = [0] * sum(col_counts)
        
        # Заполняем CSC
        for i in range(n_rows):
            for j in range(n_cols):
                if abs(matrix[i][j]) > EPS:
                    pos = next_pos[j]
                    data[pos] = float(matrix[i][j])
                    indices[pos] = i
                    next_pos[j] += 1
        
        return cls(data, indices, indptr, (n_rows, n_cols))
    
    def to_coo(self) -> 'COOMatrix':
        from COO import COOMatrix
        rows = []
        cols = []
        data = []
        
        for j in range(self.n_cols):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for idx in range(start, end):
                rows.append(self.indices[idx])
                cols.append(j)
                data.append(self.data[idx])
        
        return COOMatrix(rows, cols, data, self.shape)
    
    def to_csr(self) -> 'CSRMatrix':
        return self.transpose()
    
    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        # Через dense для простоты
        dense_self = self.to_dense()
        dense_other = other.to_dense()
        return CSCMatrix.from_dense([
            [dense_self[i][j] + dense_other[i][j] for j in range(self.n_cols)]
            for i in range(self.n_rows)
        ])
    
    def _mul_impl(self, scalar: float) -> 'Matrix':
        new_data = [val * scalar for val in self.data]
        return CSCMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)
    
    def transpose(self) -> 'Matrix':
        from CSR import CSRMatrix
        
        n_rows, n_cols = self.shape
        nnz = len(self.data)
        
        # Подсчитываем количество элементов в каждой строке
        row_counts = [0] * n_rows
        for row in self.indices:
            row_counts[row] += 1
        
        # Вычисляем indptr для CSR
        csr_indptr = [0] * (n_rows + 1)
        for i in range(n_rows):
            csr_indptr[i + 1] = csr_indptr[i] + row_counts[i]
        
        # Рабочий массив для заполнения
        next_pos = csr_indptr.copy()
        csr_data = [0.0] * nnz
        csr_indices = [0] * nnz
        
        # Заполняем CSR
        for j in range(n_cols):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for idx in range(start, end):
                i = self.indices[idx]
                pos = next_pos[i]
                csr_data[pos] = self.data[idx]
                csr_indices[pos] = j
                next_pos[i] += 1
        
        return CSRMatrix(csr_data, csr_indices, csr_indptr, (n_cols, n_rows))
    
    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        # Через dense для простоты
        dense_self = self.to_dense()
        dense_other = other.to_dense()
        
        n_rows = self.n_rows
        n_cols = other.shape[1]
        n_inner = self.n_cols
        
        result_dense = [[0.0] * n_cols for _ in range(n_rows)]
        
        for i in range(n_rows):
            for j in range(n_cols):
                sum_val = 0.0
                for k in range(n_inner):
                    sum_val += dense_self[i][k] * dense_other[k][j]
                result_dense[i][j] = sum_val
        
        return CSCMatrix.from_dense(result_dense)
    
    def __str__(self) -> str:
        return f"CSCMatrix(shape={self.shape}, nnz={len(self.data)})"