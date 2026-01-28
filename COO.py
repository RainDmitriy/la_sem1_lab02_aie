from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix
from typing import Dict, Tuple, List


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        if not (len(data) == len(row) == len(col)):
            raise ValueError("Длины data, row, col не равны")
        self.data = data
        self.row = row
        self.col = col

    def to_dense(self) -> DenseMatrix:
        '''Преобразует COO в плотную матрицу.'''
        rows, cols = self.shape
        dense = [[0] * cols for _ in range(rows)]
        for i in range(len(self.data)):
            dense[self.row[i]][self.col[i]] = self.data[i]
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        '''Сложение COO матриц.'''
        if isinstance(other, COOMatrix):
            sum_dict: Dict[Tuple[int, int], float] = {}

            for value, r, c in zip(self.data, self.row, self.col):
                sum_dict[(r, c)] = sum_dict.get((r, c), 0.0) + value

            for value, r, c in zip(other.data, other.row, other.col):
                sum_dict[(r, c)] = sum_dict.get((r, c), 0.0) + value

            new_data, new_row, new_col = [], [], []
            for (r, c), val in sum_dict.items():
                if val != 0:
                    new_data.append(val)
                    new_row.append(r)
                    new_col.append(c)

            return COOMatrix(new_data, new_row, new_col, self.shape)
        else:
            from CSR import CSRMatrix
            csr_self = self._to_csr()
            csr_other = CSRMatrix.from_dense(other.to_dense())
            return csr_self._add_impl(csr_other)._to_coo()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        '''Умножение COO на скаляр.'''
        new_data = [elem * scalar for elem in self.data]
        return COOMatrix(new_data, self.row[:], self.col[:], self.shape)

    def transpose(self) -> 'Matrix':
        '''Транспонирование COO матрицы.'''
        new_shape = (self.shape[1], self.shape[0])
        return COOMatrix(self.data[:], self.col[:], self.row[:], new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        '''Умножение COO матриц.'''
        from CSR import CSRMatrix

        if self.shape[1] != other.shape[0]:
            raise ValueError(f"неправильные размеры матриц")

        csr_self = self._to_csr()

        if isinstance(other, COOMatrix):
            csr_other = other._to_csr()
        else:
            csr_other = CSRMatrix.from_dense(other.to_dense())

        result_csr = csr_self._matmul_impl(csr_other)
        return result_csr._to_coo()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        '''Создание COO из плотной матрицы.'''
        data, row, col = [], [], []

        for i in range(len(dense_matrix)):
            for j in range(len(dense_matrix[0])):
                value = dense_matrix[i][j]
                if value != 0:
                    data.append(value)
                    row.append(i)
                    col.append(j)

        return cls(data, row, col, (len(dense_matrix), len(dense_matrix[0])))

    def _to_csc(self) -> 'CSCMatrix':
        '''
        Преобразование COOMatrix в CSCMatrix.
        '''
        from CSC import CSCMatrix

        if len(self.data) == 0:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)

        elements = list(zip(self.col, self.row, self.data))
        elements.sort()

        data = [elem[2] for elem in elements]
        indices = [elem[1] for elem in elements]
        indptr = [0] * (self.shape[1] + 1)

        for col_idx, _, _ in elements:
            indptr[col_idx + 1] += 1

        for j in range(self.shape[1]):
            indptr[j + 1] += indptr[j]

        return CSCMatrix(data, indices, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        '''
        Преобразование COOMatrix в CSRMatrix.
        '''
        from CSR import CSRMatrix

        if len(self.data) == 0:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)

        elements = list(zip(self.row, self.col, self.data))
        elements.sort()

        data = [elem[2] for elem in elements]
        indices = [elem[1] for elem in elements]
        indptr = [0] * (self.shape[0] + 1)

        for row_idx, _, _ in elements:
            indptr[row_idx + 1] += 1

        for i in range(self.shape[0]):
            indptr[i + 1] += indptr[i]

        return CSRMatrix(data, indices, indptr, self.shape)