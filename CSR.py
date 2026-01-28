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
        data, row, col = [], [], []

        for r in range(rows):
            for idx in range(self.indptr[r], self.indptr[r + 1]):
                c = self.indices[idx]
                v = self.data[idx]
                row.append(c)  #менянм row и col
                col.append(r)
                data.append(v)

        elements = list(zip(row, col, data))
        elements.sort(key=lambda x: (x[0], x[1]))

        sorted_col = [e[0] for e in elements]
        sorted_row = [e[1] for e in elements]
        sorted_data = [e[2] for e in elements]

        new_shape = (cols, rows)
        new_cols = cols
        indptr = [0] * (new_cols + 1)
        for c in sorted_col:
            indptr[c + 1] += 1
        for i in range(1, new_cols + 1):
            indptr[i] += indptr[i - 1]

        return CSCMatrix(sorted_data, sorted_row, indptr, new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        from COO import COOMatrix
        from CSC import CSCMatrix

        a_coo = self._to_coo()
        if isinstance(other, CSCMatrix):
            b_coo = other._to_coo()
        else:
            b_coo = other._to_coo()

        result_dict = {}
        for v1, r1, c1 in zip(a_coo.data, a_coo.row, a_coo.col):
            for v2, r2, c2 in zip(b_coo.data, b_coo.row, b_coo.col):
                if c1 == r2:
                    result_dict[(r1, c2)] = result_dict.get((r1, c2), 0) + v1 * v2

        #собираем COO
        data, row, col = [], [], []
        for r, c in sorted(result_dict.keys(), key=lambda x: (x[0], x[1])):  # сорт
            v = result_dict[(r, c)]
            if v != 0:
                data.append(v)
                row.append(r)
                col.append(c)

        shape = (self.shape[0], other.shape[1])
        coo_result = COOMatrix(data, row, col, shape)

        # в томже формате ретерн
        return coo_result._to_csr()

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
