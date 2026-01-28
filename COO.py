from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix
from typing import Dict, Tuple, List


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.data = data.copy()
        self.row = row.copy()
        self.col = col.copy()
        self.nnz = len(data)

    def to_dense(self) -> DenseMatrix:
        '''Преобразует разреженную матрицу в плотную.'''
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]
        for val, r, c in zip(self.data, self.row, self.col):
            dense[r][c] = val
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        '''Реализация сложения с другой матрицей.'''
        if isinstance(other, COOMatrix):
            sum_dict: Dict[Tuple[int, int], float] = {}

            for i in range(len(self.data)):
                key = (self.row[i], self.col[i])
                sum_dict[key] = sum_dict.get(key, 0.0) + self.data[i]

            for i in range(len(other.data)):
                key = (other.row[i], other.col[i])
                sum_dict[key] = sum_dict.get(key, 0.0) + other.data[i]

            new_data, new_row, new_col = [], [], []
            for (r, c), val in sum_dict.items():
                if abs(val) > 1e-12:
                    new_data.append(val)
                    new_row.append(r)
                    new_col.append(c)

            sorted_elements = sorted(zip(new_row, new_col, new_data))
            if sorted_elements:
                new_row, new_col, new_data = zip(*sorted_elements)
            else:
                new_row, new_col, new_data = [], [], []

            return COOMatrix(list(new_data), list(new_row), list(new_col), self.shape)
        else:
            from CSR import CSRMatrix
            csr_self = self._to_csr()
            csr_other = CSRMatrix.from_dense(other.to_dense())
            return csr_self._add_impl(csr_other)._to_coo()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        '''Реализация умножения на скаляр.'''
        if abs(scalar) < 1e-12:
            return COOMatrix([], [], [], self.shape)
        new_data = [val * scalar for val in self.data]
        return COOMatrix(new_data, self.row.copy(), self.col.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        '''Транспонирование матрицы.'''
        new_row = self.col.copy()
        new_col = self.row.copy()
        new_data = self.data.copy()

        sorted_elements = sorted(zip(new_row, new_col, new_data))
        if sorted_elements:
            new_row, new_col, new_data = zip(*sorted_elements)
        else:
            new_row, new_col, new_data = [], [], []

        return COOMatrix(list(new_data), list(new_row), list(new_col),
                         (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        '''Реализация умножения матриц.'''
        from CSR import CSRMatrix

        if self.shape[1] != other.shape[0]:
            raise ValueError("Несовместимые размерности для умножения")

        if isinstance(other, COOMatrix):
            other_csr = other._to_csr()
        else:
            other_csr = CSRMatrix.from_dense(other.to_dense())

        rows_A, cols_A = self.shape
        rows_B, cols_B = other.shape

        csr_self = self._to_csr()
        result_csr = csr_self._matmul_impl(other_csr)
        return result_csr._to_coo()

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

        elements = list(zip(self.col, self.row, self.data))
        elements.sort(key=lambda x: (x[0], x[1]))

        data = [elem[2] for elem in elements]
        indices = [elem[1] for elem in elements]
        indptr = [0] * (self.shape[1] + 1)

        for col, _, _ in elements:
            indptr[col + 1] += 1

        for j in range(1, self.shape[1] + 1):
            indptr[j] += indptr[j - 1]

        return CSCMatrix(data, indices, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        '''
        Преобразование COOMatrix в CSRMatrix.
        '''
        from CSR import CSRMatrix

        if self.nnz == 0:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)

        elements = list(zip(self.row, self.col, self.data))
        elements.sort(key=lambda x: (x[0], x[1]))

        data = [elem[2] for elem in elements]
        indices = [elem[1] for elem in elements]
        indptr = [0] * (self.shape[0] + 1)

        for row, _, _ in elements:
            indptr[row + 1] += 1

        for i in range(1, self.shape[0] + 1):
            indptr[i] += indptr[i - 1]

        return CSRMatrix(data, indices, indptr, self.shape)