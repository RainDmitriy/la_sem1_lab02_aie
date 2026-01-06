from base import Matrix
from types import COOData, COORows, COOCols, Shape, DenseMatrix


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.data = list(data)
        self.rows = list(row)
        self.cols = list(col)
        self.nnz = len(data)

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу."""
        n, m = self.shape
        mat = [[0]*m for _ in range(n)]
        for i in range(self.nnz):
            r = self.rows[i]
            c = self.cols[i]
            mat[r][c] = self.data[i]
        return mat

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц."""
        if not isinstance(other, COOMatrix):
            other_dense = other.to_dense()
        else:
            other_dense = other.to_dense()
        
        self_dense = self.to_dense()
        n, m = self.shape

        result = []
        for i in range(n):
            row = []
            for j in range(m):
                row.append(self_dense[i][j] + other_dense[i][j])
            result.append(row)
        
        return COOMatrix.from_dense(result)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        new_data = [x * scalar for x in self.data]
        return COOMatrix(new_data, self.rows, self.cols, self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        new_shape = (self.shape[1], self.shape[0])
        return COOMatrix(self.data, self.cols, self.rows, new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""
        A_dense = self.to_dense()
        B_dense = other.to_dense()
        
        rows_A, cols_A = self.shape
        rows_B, cols_B = other.shape
        
        result = [[0.0] * cols_B for _ in range(rows_A)]
        
        for i in range(rows_A):
            for k in range(cols_A):
                if abs(A_dense[i][k]) > 1e-15:
                    for j in range(cols_B):
                        result[i][j] += A_dense[i][k] * B_dense[k][j]
        
        return COOMatrix.from_dense(result)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        
        data = []
        row_indices = []
        col_indices = []
        
        for i in range(rows):
            for j in range(cols):
                val = dense_matrix[i][j]
                if abs(val) > 1e-15:
                    data.append(val)
                    row_indices.append(i)
                    col_indices.append(j)
        
        return cls(data, row_indices, col_indices, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix

        if not self.data:
            return CSCMatrix([], [], [0]*(self.shape[1]+1), self.shape)

        items = list(zip(self.cols, self.rows, self.data))
        items.sort(key=lambda x: (x[0], x[1]))
        
        data = [d for _, _, d in items]
        indices = [r for _, r, _ in items]
        _, m = self.shape
        indptr = [0]*(m+1)
        
        for c, _, _ in items:
            indptr[c+1] += 1
        
        for j in range(m):
            indptr[j+1] += indptr[j]
        
        return CSCMatrix(data, indices, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix

        if not self.data:
            return CSRMatrix([], [], [0]*(self.shape[0]+1), self.shape)

        items = list(zip(self.rows, self.cols, self.data))
        items.sort(key=lambda x: (x[0], x[1]))
        
        data = [d for _, _, d in items]
        indices = [c for _, c, _ in items]
        n, _ = self.shape
        indptr = [0]*(n+1)
        
        for r, _, _ in items:
            indptr[r+1] += 1

        for i in range(n):
            indptr[i+1] += indptr[i]
        
        return CSRMatrix(data, indices, indptr, self.shape)
