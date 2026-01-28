from base import Matrix
from collections import defaultdict
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix

class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]

        for col in range(cols):
            start = self.indptr[col]
            end = self.indptr[col + 1]
            for idx in range(start, end):
                dense[self.indices[idx]][col] = self.data[idx]

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        if self.shape != other.shape:
            raise ValueError("матрицы разного размера")

        return self._to_coo()._add_impl(other._to_coo())._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        if scalar == 0:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)

        new_data = [val * scalar for val in self.data]

        return CSCMatrix(new_data, self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Hint:
        Результат - в CSR формате (с теми же данными, но с интерпретацией строк как столбцов).
        """
        from CSR import CSRMatrix

        rows, cols = self.shape
        data, row, col = [], [], []

        for c in range(cols):
            for idx in range(self.indptr[c], self.indptr[c + 1]):
                r = self.indices[idx]
                v = self.data[idx]
                row.append(c)  # меняем row и col
                col.append(r)
                data.append(v)

        elements = list(zip(row, col, data))
        elements.sort(key=lambda x: (x[0], x[1]))  # сортировка по row, потом col

        sorted_row = [e[0] for e in elements]
        sorted_col = [e[1] for e in elements]
        sorted_data = [e[2] for e in elements]

        new_shape = (cols, rows)
        new_rows = cols
        indptr = [0] * (new_rows + 1)
        for r in sorted_row:
            indptr[r + 1] += 1
        for i in range(1, new_rows + 1):
            indptr[i] += indptr[i - 1]

        return CSRMatrix(sorted_data, sorted_col, indptr, new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        from COO import COOMatrix
        a_coo = self._to_coo()
        b_coo = other._to_coo()

        result_dict = {}
        for v1, r1, c1 in zip(a_coo.data, a_coo.row, a_coo.col):
            for v2, r2, c2 in zip(b_coo.data, b_coo.row, b_coo.col):
                if c1 == r2:
                    result_dict[(r1, c2)] = result_dict.get((r1, c2), 0) + v1 * v2

        data, row, col = [], [], []
        for r, c in sorted(result_dict.keys(), key=lambda x: (x[1], x[0])):  # сортировка по col, затем row
            v = result_dict[(r, c)]
            if v != 0:
                data.append(v)
                row.append(r)
                col.append(c)

        shape = (self.shape[0], other.shape[1])
        return COOMatrix(data, row, col, shape)._to_csc()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0

        data = []
        indices = []
        indptr = [0]

        for j in range(cols):
            count = 0
            for i in range(rows):
                val = dense_matrix[i][j]
                if val != 0:
                    data.append(val)
                    indices.append(i)
                    count += 1
            indptr.append(indptr[-1] + count)

        return cls(data, indices, indptr, (rows, cols))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix

        coo = self._to_coo()
        elements = list(zip(coo.row, coo.col, coo.data))
        elements.sort(key=lambda x: (x[0], x[1]))

        sorted_row = [e[0] for e in elements]
        sorted_col = [e[1] for e in elements]
        sorted_data = [e[2] for e in elements]

        rows = self.shape[0]
        indptr = [0] * (rows + 1)
        for r in sorted_row:
            indptr[r + 1] += 1
        for i in range(1, rows + 1):
            indptr[i] += indptr[i - 1]

        return CSRMatrix(sorted_data, sorted_col, indptr, self.shape)

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        from COO import COOMatrix
        data, row, col = [], [], []
        cols = self.shape[1]
        for j in range(cols):
            for idx in range(self.indptr[j], self.indptr[j + 1]):
                row.append(self.indices[idx])
                col.append(j)
                data.append(self.data[idx])
        return COOMatrix(data, row, col, self.shape)