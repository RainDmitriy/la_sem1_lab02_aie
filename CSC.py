from base import Matrix
from types import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу."""
        dense = []
        rows, cols = self.shape

        for _ in range(rows):
            dense.append([0.0] * cols)

        for j in range(cols):
            start = self.indptr[j]
            end = self.indptr[j + 1]

            for k in range(start, end):
                row_index = self.indices[k]
                dense[row_index][j] = self.data[k]

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        self_coo = self._to_coo()
        other_coo = other._to_coo()
        result_coo = self_coo + other_coo
        return result_coo._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        new_data = [x * scalar for x in self.data]
        return CSCMatrix(new_data, list(self.indices), list(self.indptr), self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Hint:
        Результат - в CSR формате (с теми же данными, но с интерпретацией строк как столбцов).
        """
        from CSR import CSRMatrix
        return CSRMatrix(
            list(self.data),
            list(self.indices),
            list(self.indptr),
            (self.shape[1], self.shape[0])
        )

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        dense_self = self.to_dense()
        dense_other = other.to_dense()

        rows_a = len(dense_self)
        cols_a = len(dense_self[0])
        cols_b = len(dense_other[0])

        result_dense = []
        for i in range(rows_a):
            new_row = []
            for j in range(cols_b):
                sum_val = 0.0
                for k in range(cols_a):
                    sum_val += dense_self[i][k] * dense_other[k][j]
                new_row.append(sum_val)
            result_dense.append(new_row)

        return CSCMatrix.from_dense(result_dense)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        data = []
        indices = []
        indptr = [0]

        rows = len(dense_matrix)
        cols = len(dense_matrix[0])
        cur_idx = 0

        for j in range(cols):
            for i in range(rows):
                val = dense_matrix[i][j]
                if val != 0:
                    data.append(val)
                    indices.append(i)
                    cur_idx += 1
            indptr.append(cur_idx)

        return cls(data, indices, indptr, (rows, cols))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        return self._to_coo()._to_csr()

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        from COO import COOMatrix

        coo_rows = list(self.indices)
        coo_data = list(self.data)
        coo_cols = []

        for j in range(len(self.indptr) - 1):
            count = self.indptr[j + 1] - self.indptr[j]
            for _ in range(count):
                coo_cols.append(j)

        return COOMatrix(coo_data, coo_rows, coo_cols, self.shape)
