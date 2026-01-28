from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix
from typing import Dict, Tuple, List


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.row = row
        self.col = col
        self.nnz = len(data)

    def to_dense(self) -> DenseMatrix:
        '''Преобразует COO в плотную матрицу.'''
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]
        for idx in range(self.nnz):
            dense[self.row[idx]][self.col[idx]] = self.data[idx]
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        '''Сложение COO матриц.'''
        if not isinstance(other, COOMatrix):
            from CSR import CSRMatrix
            return self._to_csr()._add_impl(CSRMatrix.from_dense(other.to_dense()))._to_coo()

        sum_dict: Dict[Tuple[int, int], float] = {}

        for idx in range(self.nnz):
            key = (self.row[idx], self.col[idx])
            sum_dict[key] = sum_dict.get(key, 0.0) + self.data[idx]

        for idx in range(other.nnz):
            key = (other.row[idx], other.col[idx])
            sum_dict[key] = sum_dict.get(key, 0.0) + other.data[idx]

        new_data, new_row, new_col = [], [], []
        for (r, c), val in sum_dict.items():
            if abs(val) > 1e-12:
                new_data.append(val)
                new_row.append(r)
                new_col.append(c)

        return COOMatrix(new_data, new_row, new_col, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        '''Умножение COO на скаляр.'''
        if abs(scalar) < 1e-12:
            return COOMatrix([], [], [], self.shape)
        new_data = [val * scalar for val in self.data]
        return COOMatrix(new_data, self.row, self.col, self.shape)

    def transpose(self) -> 'Matrix':
        '''Транспонирование COO матрицы.'''
        return COOMatrix(self.data, self.col, self.row, (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        '''Умножение COO матриц.'''
        from CSR import CSRMatrix

        if self.shape[1] != other.shape[0]:
            raise ValueError("Несовместимые размерности для умножения")

        csr_self = self._to_csr()
        if isinstance(other, COOMatrix):
            csr_other = other._to_csr()
        else:
            csr_other = CSRMatrix.from_dense(other.to_dense())

        return csr_self._matmul_impl(csr_other)._to_coo()

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

        if self.nnz == 0:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)

        elements = []
        for idx in range(self.nnz):
            elements.append((self.col[idx], self.row[idx], self.data[idx]))

        elements.sort()

        data = [elem[2] for elem in elements]
        indices = [elem[1] for elem in elements]
        indptr = [0] * (self.shape[1] + 1)

        for col, _, _ in elements:
            indptr[col + 1] += 1

        for j in range(self.shape[1]):
            indptr[j + 1] += indptr[j]

        return CSCMatrix(data, indices, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        '''
        Преобразование COOMatrix в CSRMatrix.
        '''
        from CSR import CSRMatrix

        if self.nnz == 0:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)

        elements = []
        for idx in range(self.nnz):
            elements.append((self.row[idx], self.col[idx], self.data[idx]))

        elements.sort()

        data = [elem[2] for elem in elements]
        indices = [elem[1] for elem in elements]
        indptr = [0] * (self.shape[0] + 1)

        for row, _, _ in elements:
            indptr[row + 1] += 1

        for i in range(self.shape[0]):
            indptr[i + 1] += indptr[i]

        return CSRMatrix(data, indices, indptr, self.shape)