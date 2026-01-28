from base import Matrix
from types import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix
from COO import COOMatrix
from CSR import CSRMatrix

class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        rows, cols = shape

        if len(indptr) != cols + 1:
            raise ValueError("Длина indptr должна быть cols + 1")

        if len(data) != len(indices):
            raise ValueError("Длины data и indices должны совпадать")

        if indptr[-1] != len(data):
            raise ValueError("Последний элемент indptr должен равняться len(data)")

        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0 for _ in range(cols)] for _ in range(rows)]

        for j in range(cols):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for k in range(start, end):
                i = self.indices[k]
                dense[i][j] += self.data[k]   

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        if not isinstance(other, CSCMatrix):
            other = other._to_csc()

        plus_data, plus_indices, plus_indptr = [], [], [0]
        cols = self.shape[1]

        for j in range(cols):
            col_sum = {}

            for k in range(self.indptr[j], self.indptr[j + 1]):
                i = self.indices[k]
                col_sum[i] = col_sum.get(i, 0) + self.data[k]

            for k in range(other.indptr[j], other.indptr[j + 1]):
                i = other.indices[k]
                col_sum[i] = col_sum.get(i, 0) + other.data[k]

            for i in sorted(col_sum):
                v = col_sum[i]
                if v != 0:
                    plus_indices.append(i)
                    plus_data.append(v)

            plus_indptr.append(len(plus_data))

        return CSCMatrix(plus_data, plus_indices, plus_indptr, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        new_data = [v * scalar for v in self.data]
        return CSCMatrix(new_data, list(self.indices), list(self.indptr), self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Hint:
        Результат - в CSR формате (с теми же данными, но с интерпретацией строк как столбцов).
        """
        return CSRMatrix(
            self.data,
            self.indices,
            self.indptr,
            (self.shape[1], self.shape[0])
        )

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        if self.shape[1] != other.shape[0]:
            raise ValueError("Несовместимые размеры матриц")

        return self._to_csr()._matmul_impl(other._to_csr())

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        data, indices, indptr = [], [], [0]

        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0

        for j in range(cols):
            for i in range(rows):
                v = dense_matrix[i][j]
                if v != 0:
                    data.append(v)
                    indices.append(i)
            indptr.append(len(data))

        return cls(data, indices, indptr, (rows, cols))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        rows, cols = self.shape
        nnz = len(self.data)

        row_count = [0] * rows
        for i in self.indices:
            row_count[i] += 1

        indptr = [0] * (rows + 1)
        for i in range(rows):
            indptr[i + 1] = indptr[i] + row_count[i]

        data = [0] * nnz
        indices = [0] * nnz
        current = indptr.copy()

        for j in range(cols):
            for k in range(self.indptr[j], self.indptr[j + 1]):
                i = self.indices[k]
                pos = current[i]
                data[pos] = self.data[k]
                indices[pos] = j
                current[i] += 1

        return CSRMatrix(data, indices, indptr, (rows, cols))

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        rows, cols = self.shape
        data, row, col = [], [], []

        for j in range(cols):
            for k in range(self.indptr[j], self.indptr[j + 1]):
                i = self.indices[k]
                data.append(self.data[k])
                row.append(i)
                col.append(j)

        return COOMatrix(data, row, col, self.shape)
