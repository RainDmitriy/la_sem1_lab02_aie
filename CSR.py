from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix
from CSC import CSCMatrix
from COO import COOMatrix

class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        rows, cols = self.shape
        matrix = [[0.0 for _ in range(cols)] for _ in range(rows)]

        for row in range(rows):
            row_start = self.indptr[row]
            row_end = self.indptr[row + 1]
            for i in range(row_start, row_end):
                col = self.indices[i]
                value = self.data[i]
                matrix[row][col] += value
        return matrix

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        rows, cols = self.shape
        if rows != other.shape[0] or cols != other.shape[1]:
            raise ValueError("Матрицы должны быть одной размерности")
        if not isinstance(other, CSRMatrix):
            raise TypeError("Матрица должна быть CSR")

        data = []
        indices = []
        indptr = [0]
        for i in range(rows):
            row = {}
            for j in range(self.indptr[i], self.indptr[i + 1]):
                col = self.indices[j]
                row[col] = row.get(col, 0.0) + self.data[j]
            for j in range(other.indptr[i], other.indptr[i + 1]):
                col = other.indices[j]
                row[col] = row.get(col, 0.0) + other.data[j]

            for col in sorted(row):
                value = row[col]
                if value != 0:
                    data.append(value)
                    indices.append(col)
            indptr.append(len(data))

        return CSRMatrix(data, indices, indptr, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        values = [x * scalar for x in self.data]
        return CSRMatrix(values, self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Hint:
        Результат - в CSC формате (с теми же данными, но с интерпретацией столбцов как строк).
        """
        return CSCMatrix(self.data, self.indices, self.indptr, (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        if self.shape[1] != other.shape[0]:
            raise ValueError("Умножение матриц невозможно")
        if not isinstance(other, CSRMatrix):
            raise TypeError("Матрица должна быть CSR")
        a = self
        b = other
        rows_a = a.shape[0]
        cols_b = b.shape[1]
        data = []
        indices = []
        indptr = [0]

        for row in range(rows_a):
            result = {}
            for i_a in range(a.indptr[row], a.indptr[row + 1]):
                l = a.indices[i_a]
                value_a = a.data[i_a]
                for i_b in range(b.indptr[l], b.indptr[l + 1]):
                    j = b.indices[i_b]
                    value_b = b.data[i_b]

                    if j not in result:
                        result[j] = 0.0
                    result[j] += value_a * value_b

            for col in sorted(result.keys()):
                value = result[col]
                if value != 0:
                    indices.append(col)
                    data.append(value)
            indptr.append(len(data))

        return CSRMatrix(data, indices, indptr, (rows_a, cols_b))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        rows_num = len(dense_matrix)
        cols_num = len(dense_matrix[0])
        data = []
        indices = []
        indptr = [0]

        for r in range(rows_num):
            for c in range(cols_num):
                value = dense_matrix[r][c]
                if value != 0:
                    indices.append(c)
                    data.append(value)
            indptr.append(len(data))

        return cls(data, indices, indptr, (rows_num, cols_num))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSRMatrix в CSCMatrix.
        """
        if not isinstance(self, CSRMatrix):
            raise TypeError("Матрица должна быть CSR")
        coo = self._to_coo()
        return coo._to_csc()
    
    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSRMatrix в COOMatrix.
        """
        cols = self.indices
        rows = []
        for row in range(self.shape[0]):
            count = self.indptr[row + 1] - self.indptr[row]
            rows.extend([row] * count)

        return COOMatrix(self.data, rows, cols, self.shape)