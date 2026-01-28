from base import Matrix
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.nnz = len(data)

    def to_dense(self) -> DenseMatrix:
        '''Преобразует CSC в плотную матрицу.'''
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]

        for j in range(cols):
            start, end = self.indptr[j], self.indptr[j + 1]
            for idx in range(start, end):
                i = self.indices[idx]
                dense[i][j] = self.data[idx]

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        '''Сложение CSC матриц.'''
        from COO import COOMatrix

        return self._to_coo()._add_impl(other)._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        '''Умножение CSC на скаляр.'''
        if abs(scalar) < 1e-12:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)

        new_data = [val * scalar for val in self.data]
        return CSCMatrix(new_data, self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        '''
        Транспонирование CSC матрицы.
        Hint:
        Результат - в CSR формате (с теми же данными, но с интерпретацией строк как столбцов).
        '''
        from CSR import CSRMatrix

        rows, cols = self.shape

        row_counts = [0] * rows
        for row in self.indices:
            row_counts[row] += 1

        indptr = [0] * (rows + 1)
        for i in range(rows):
            indptr[i + 1] = indptr[i] + row_counts[i]

        data = [0.0] * self.nnz
        indices = [0] * self.nnz
        row_positions = indptr[:]

        for j in range(cols):
            start, end = self.indptr[j], self.indptr[j + 1]
            for idx in range(start, end):
                i = self.indices[idx]
                pos = row_positions[i]
                data[pos] = self.data[idx]
                indices[pos] = j
                row_positions[i] += 1

        return CSRMatrix(data, indices, indptr, (cols, rows))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        '''Умножение CSC матриц.'''
        from CSR import CSRMatrix

        return self._to_csr()._matmul_impl(other)._to_csc()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        '''Создание CSC из плотной матрицы.'''
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        data, indices = [], []
        indptr = [0]

        for j in range(cols):
            for i in range(rows):
                val = dense_matrix[i][j]
                if abs(val) > 1e-12:
                    data.append(val)
                    indices.append(i)
            indptr.append(len(data))

        return cls(data, indices, indptr, (rows, cols))

    def _to_csr(self) -> 'CSRMatrix':
        '''
        Преобразование CSCMatrix в CSRMatrix.
        '''
        return self.transpose()

    def _to_coo(self) -> 'COOMatrix':
        '''
        Преобразование CSCMatrix в COOMatrix.
        '''
        from COO import COOMatrix

        rows, cols = self.shape
        data, row_indices, col_indices = [], [], []

        for j in range(cols):
            start, end = self.indptr[j], self.indptr[j + 1]
            for idx in range(start, end):
                data.append(self.data[idx])
                row_indices.append(self.indices[idx])
                col_indices.append(j)

        return COOMatrix(data, row_indices, col_indices, self.shape)