from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.data = data.copy()
        self.row = row.copy()
        self.col = col.copy()

    def to_dense(self) -> DenseMatrix:
        '''Преобразует COO в плотную матрицу.'''
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]
        for val, r, c in zip(self.data, self.row, self.col):
            dense[r][c] = val
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        '''Сложение COO матриц.'''
        dense_other = other.to_dense()
        dense_self = self.to_dense()
        rows, cols = self.shape
        result = [[0.0] * cols for _ in range(rows)]

        for i in range(rows):
            for j in range(cols):
                result[i][j] = dense_self[i][j] + dense_other[i][j]

        return COOMatrix.from_dense(result)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        '''Умножение COO на скаляр.'''
        if abs(scalar) < 1e-12:
            return COOMatrix([], [], [], self.shape)
        new_data = [val * scalar for val in self.data]
        return COOMatrix(new_data, self.row.copy(), self.col.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        '''Транспонирование COO матрицы.'''
        return COOMatrix(self.data.copy(), self.col.copy(), self.row.copy(),
                         (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        '''Умножение COO матриц.'''
        dense_self = self.to_dense()
        dense_other = other.to_dense()
        rows_A, cols_A = self.shape
        rows_B, cols_B = other.shape

        result = [[0.0] * cols_B for _ in range(rows_A)]

        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    result[i][j] += dense_self[i][k] * dense_other[k][j]

        return COOMatrix.from_dense(result)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        '''Создание COO из плотной матрицы.'''
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        data, row_indices, col_indices = [], [], []

        for i in range(rows):
            for j in range(cols):
                val = dense_matrix[i][j]
                if abs(val) > 1e-12:
                    data.append(val)
                    row_indices.append(i)
                    col_indices.append(j)

        return cls(data, row_indices, col_indices, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        '''
        Преобразование COOMatrix в CSCMatrix.
        '''
        from CSC import CSCMatrix
        return CSCMatrix.from_dense(self.to_dense())

    def _to_csr(self) -> 'CSRMatrix':
        '''
        Преобразование COOMatrix в CSRMatrix.
        '''
        from CSR import CSRMatrix
        return CSRMatrix.from_dense(self.to_dense())