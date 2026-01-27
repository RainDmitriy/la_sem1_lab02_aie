from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data.copy()
        self.indices = indices.copy()
        self.indptr = indptr.copy()

    def to_dense(self) -> DenseMatrix:
        '''Преобразует CSR в плотную матрицу.'''
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]
        for i in range(rows):
            for idx in range(self.indptr[i], self.indptr[i + 1]):
                j = self.indices[idx]
                dense[i][j] = self.data[idx]
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        '''Сложение CSR матриц.'''
        from COO import COOMatrix
        coo_self = self._to_coo()
        coo_other = COOMatrix.from_dense(other.to_dense())
        result_coo = coo_self._add_impl(coo_other)
        return result_coo._to_csr()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        '''Умножение CSR на скаляр.'''
        if abs(scalar) < 1e-12:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)
        new_data = [val * scalar for val in self.data]
        return CSRMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        '''
        Транспонирование CSR матрицы.
        Hint:
        Результат - в CSC формате (с теми же данными, но с интерпретацией столбцов как строк).
        '''
        from COO import COOMatrix
        return self._to_coo().transpose()._to_csc()

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        '''Умножение CSR матриц.'''
        from COO import COOMatrix
        coo_self = self._to_coo()
        coo_other = COOMatrix.from_dense(other.to_dense())
        result_coo = coo_self._matmul_impl(coo_other)
        return result_coo._to_csr()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        '''Создание CSR из плотной матрицы.'''
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        data, indices = [], []
        indptr = [0]

        for i in range(rows):
            for j in range(cols):
                val = dense_matrix[i][j]
                if abs(val) > 1e-12:
                    data.append(val)
                    indices.append(j)
            indptr.append(len(data))

        return cls(data, indices, indptr, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        '''
        Преобразование CSRMatrix в CSCMatrix.
        '''
        from CSC import CSCMatrix
        return self.transpose()

    def _to_coo(self) -> 'COOMatrix':
        '''
        Преобразование CSRMatrix в COOMatrix.
        '''
        from COO import COOMatrix
        rows, cols = self.shape
        data, row_indices, col_indices = [], [], []

        for i in range(rows):
            for idx in range(self.indptr[i], self.indptr[i + 1]):
                data.append(self.data[idx])
                row_indices.append(i)
                col_indices.append(self.indices[idx])

        return COOMatrix(data, row_indices, col_indices, self.shape)