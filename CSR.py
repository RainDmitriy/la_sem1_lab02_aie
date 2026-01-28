from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix
from collections import defaultdict

class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]

        for row in range(rows):
            start = self.indptr[row]
            end = self.indptr[row + 1]
            for idx in range(start, end):
                col = self.indices[idx]
                dense[row][col] = self.data[idx]

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        return self._to_coo()._add_impl(other._to_coo())._to_csr()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        if scalar == 0:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)

        new_data = [v * scalar for v in self.data]
        return CSRMatrix(new_data, self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Hint:
        Результат - в CSC формате (с теми же данными, но с интерпретацией столбцов как строк).
        """
        from CSC import CSCMatrix

        rows, cols = self.shape

        data = self.data[:]
        row = []
        col = []

        for r in range(rows):
            start = self.indptr[r]
            end = self.indptr[r + 1]
            for idx in range(start, end):
                row.append(r)
                col.append(self.indices[idx])

        elements = list(zip(col, row, data))  # меняем местами
        elements.sort(key=lambda x: (x[0], x[1]))

        if not elements:
            return CSCMatrix([], [], [0] * (cols + 1), (cols, rows))

        sorted_col = [elem[0] for elem in elements]
        sorted_row = [elem[1] for elem in elements]
        sorted_data = [elem[2] for elem in elements]

        indptr = [0] * (cols + 1)
        for col_idx in sorted_col:
            indptr[col_idx + 1] += 1
        for i in range(1, cols + 1):
            indptr[i] += indptr[i - 1]

        return CSRMatrix(sorted_data, sorted_row, indptr, (cols, rows))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        from CSR import CSRMatrix
        from CSC import CSCMatrix

        rows_a, cols_a = self.shape
        rows_b, cols_b = other.shape

        if isinstance(other, CSRMatrix):
            other_csc = other._to_csc()
            result_csc = self._to_csc()._matmul_impl(other_csc)
            return result_csc._to_csr()

        elif isinstance(other, CSCMatrix):
            result_data = []
            result_indices = []
            result_indptr = [0]

            for i in range(rows_a):
                row_vals = {}
                row_start = self.indptr[i]
                row_end = self.indptr[i + 1]

                for idx_a in range(row_start, row_end):
                    col_a = self.indices[idx_a]
                    val_a = self.data[idx_a]

                    start_b = other.indptr[col_a]
                    end_b = other.indptr[col_a + 1]
                    for idx_b in range(start_b, end_b):
                        row_b = other.indices[idx_b]
                        val_b = other.data[idx_b]
                        row_vals[row_b] = row_vals.get(row_b, 0) + val_a * val_b

                sorted_cols = sorted(row_vals.keys())
                for c in sorted_cols:
                    v = row_vals[c]
                    if v != 0:
                        result_data.append(v)
                        result_indices.append(c)
                result_indptr.append(len(result_data))

            return CSRMatrix(result_data, result_indices, result_indptr, (rows_a, cols_b))

        #другая COO
        elif 'COOMatrix' in str(type(other)):
            return self._matmul_impl(other._to_csr())

        else:
            dense_self = self.to_dense()
            dense_other = other.to_dense()
            result = [[0.0] * cols_b for _ in range(rows_a)]
            for i in range(rows_a):
                for j in range(cols_a):
                    if dense_self[i][j] != 0:
                        for k in range(cols_b):
                            if dense_other[j][k] != 0:
                                result[i][k] += dense_self[i][j] * dense_other[j][k]
            return self.__class__.from_dense(result)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        from COO import COOMatrix
        return COOMatrix.from_dense(dense_matrix)._to_csr()

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSRMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix

        coo = self._to_coo()
        elements = list(zip(coo.col, coo.row, coo.data))
        elements.sort(key=lambda x: (x[0], x[1]))  # сортировка по col, затем row

        sorted_col = [e[0] for e in elements]
        sorted_row = [e[1] for e in elements]
        sorted_data = [e[2] for e in elements]

        cols = self.shape[1]
        indptr = [0] * (cols + 1)
        for c in sorted_col:
            indptr[c + 1] += 1
        for i in range(1, cols + 1):
            indptr[i] += indptr[i - 1]

        return CSCMatrix(sorted_data, sorted_row, indptr, self.shape)

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSRMatrix в COOMatrix.
        """
        from COO import COOMatrix
        data, row, col = [], [], []
        rows = self.shape[0]
        for i in range(rows):
            for idx in range(self.indptr[i], self.indptr[i + 1]):
                row.append(i)
                col.append(self.indices[idx])
                data.append(self.data[idx])
        return COOMatrix(data, row, col, self.shape)
