from base import Matrix
from types import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix
from COO import COOMatrix
from CSC import CSCMatrix

class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        rows, cols = self.shape
        res = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            start = self.indptr[i]
            end = self.indptr[i+1]
            for k in range(start, end):
                col_idx = self.indices[k]
                val = self.data[k]
                res[i][col_idx] = val
                
        return res

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        merged = {}
        for i in range(self.shape[0]):
            for k in range(self.indptr[i], self.indptr[i+1]):
                merged[(i, self.indices[k])] = self.data[k]
        other_dense = other.to_dense()
        for r in range(other.shape[0]):
            for c in range(other.shape[1]):
                val = other_dense[r][c]
                if val != 0:
                     merged[(r, c)] = merged.get((r, c), 0.0) + val
        data, rows, cols = [], [], []
        for (r, c), val in merged.items():
            if val != 0:
                data.append(val)
                rows.append(r)
                cols.append(c)
        temp_coo = COOMatrix(data, rows, cols, self.shape)
        return temp_coo._to_csr()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        new_data = [v * scalar for v in self.data]
        return CSRMatrix(new_data, self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Hint:
        Результат - в CSC формате (с теми же данными, но с интерпретацией столбцов как строк).
        """
        new_shape = (self.shape[1], self.shape[0])
        return CSCMatrix(self.data, self.indices, self.indptr, new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        a = self.to_dense()
        b = other.to_dense()
        rows_a, cols_a = self.shape
        cols_b = other.shape[1]

        res_dense = [[0.0 for _ in range(cols_b)] for _ in range(rows_a)]
        
        for i in range(rows_a):
            for j in range(cols_b):
                val = sum(a[i][k] * b[k][j] for k in range(cols_a))
                res_dense[i][j] = val
                
        return CSRMatrix.from_dense(res_dense)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        
        data = []
        indices = []
        indptr = [0] 
        
        for r in range(rows):
            row_nnz = 0
            for c in range(cols):
                val = dense_matrix[r][c]
                if val != 0:
                    data.append(val)
                    indices.append(c)
                    row_nnz += 1
            indptr.append(indptr[-1] + row_nnz)
            
        return cls(data, indices, indptr, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSRMatrix в CSCMatrix.
        """
        return self._to_coo()._to_csc()
    
    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSRMatrix в COOMatrix.
        """
        
        coo_rows = []
        for i in range(self.shape[0]):
            count = self.indptr[i+1] - self.indptr[i]
            coo_rows.extend([i] * count)
            
        return COOMatrix(self.data, coo_rows, self.indices, self.shape)