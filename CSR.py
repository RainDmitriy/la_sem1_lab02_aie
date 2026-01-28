from typing import List, Tuple
from base import Matrix
from type import DenseMatrix, Shape, Vector, CSRData, CSRIndices, CSRIndptr, EPS


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.n_rows, self.n_cols = shape
        
        # Проверка
        if len(indptr) != self.n_rows + 1:
            raise ValueError(f"indptr должен иметь длину {self.n_rows + 1}, получено {len(indptr)}")
        
        if len(data) != len(indices):
            raise ValueError("data и indices должны иметь одинаковую длину")
    
    def to_dense(self) -> DenseMatrix:
        dense = [[0.0] * self.n_cols for _ in range(self.n_rows)]
        
        for i in range(self.n_rows):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for idx in range(start, end):
                j = self.indices[idx]
                dense[i][j] = self.data[idx]
        
        return dense
    
    @classmethod
    def from_dense(cls, matrix: DenseMatrix) -> 'CSRMatrix':
        n_rows = len(matrix)
        n_cols = len(matrix[0]) if n_rows > 0 else 0
        
        data = []
        indices = []
        indptr = [0]
        
        for i in range(n_rows):
            nnz_in_row = 0
            for j in range(n_cols):
                if abs(matrix[i][j]) > EPS:
                    data.append(float(matrix[i][j]))
                    indices.append(j)
                    nnz_in_row += 1
            indptr.append(indptr[-1] + nnz_in_row)
        
        return cls(data, indices, indptr, (n_rows, n_cols))
    
    def to_coo(self) -> 'COOMatrix':
        from COO import COOMatrix
        rows = []
        cols = []
        data = []
        
        for i in range(self.n_rows):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for idx in range(start, end):
                rows.append(i)
                cols.append(self.indices[idx])
                data.append(self.data[idx])
        
        return COOMatrix(rows, cols, data, self.shape)
    
    def to_csc(self) -> 'CSCMatrix':
        from CSC import CSCMatrix
        
        n_rows, n_cols = self.shape
        nnz = len(self.data)
        
        # Подсчитываем количество элементов в каждом столбце
        col_counts = [0] * n_cols
        for col in self.indices:
            col_counts[col] += 1
        
        # Вычисляем indptr для CSC
        csc_indptr = [0] * (n_cols + 1)
        for j in range(n_cols):
            csc_indptr[j + 1] = csc_indptr[j] + col_counts[j]
        
        # Рабочий массив для заполнения
        next_pos = csc_indptr.copy()
        csc_data = [0.0] * nnz
        csc_indices = [0] * nnz
        
        # Заполняем CSC
        for i in range(n_rows):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for idx in range(start, end):
                j = self.indices[idx]
                pos = next_pos[j]
                csc_data[pos] = self.data[idx]
                csc_indices[pos] = i
                next_pos[j] += 1
        
        return CSCMatrix(csc_data, csc_indices, csc_indptr, self.shape)
    
    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        # Через dense для простоты
        dense_self = self.to_dense()
        dense_other = other.to_dense()
        return CSRMatrix.from_dense([
            [dense_self[i][j] + dense_other[i][j] for j in range(self.n_cols)]
            for i in range(self.n_rows)
        ])
    
    def _mul_impl(self, scalar: float) -> 'Matrix':
        new_data = [val * scalar for val in self.data]
        return CSRMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)
    
    def transpose(self) -> 'Matrix':
        return self.to_csc()  # Транспонирование CSR = CSC
    
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
        
        return CSRMatrix.from_dense(result_dense)
    
    def __str__(self) -> str:
        return f"CSRMatrix(shape={self.shape}, nnz={len(self.data)})"