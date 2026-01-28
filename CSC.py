from base import Matrix
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
        matrix = [[0.0 for _ in range(cols)] for _ in range(rows)]

        for col in range(cols):
            col_start = self.indptr[col]
            col_end = self.indptr[col + 1]
            for i in range(col_start, col_end):
                row = self.indices[i]
                value = self.data[i]
                matrix[row][col] += value
        return matrix


    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        if not isinstance(other, CSCMatrix):
            raise TypeError("Неверный тип матрицы")
        if self.shape != other.shape:
            raise ValueError("Сложение возможно только при одинаковой размерности")

        cols = self.shape[1]
        indptr = [0]
        indices = []
        data = []

        for c in range(cols):
            start_a, end_a = self.indptr[c], self.indptr[c + 1]
            col_a = {self.indices[i]: self.data[i] for i in range(start_a, end_a)}

            start_b, end_b = other.indptr[c], other.indptr[c + 1]
            col_b = {other.indices[i]: other.data[i] for i in range(start_b, end_b)}

            rows = sorted(set(col_a.keys()) | set(col_b.keys()))
            for r in rows:
                value = col_a.get(r, 0.0) + col_b.get(r, 0.0)
                if value != 0:
                    indices.append(r)
                    data.append(value)
            indptr.append(len(data))

        return CSCMatrix(data, indices, indptr, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        values = [x * scalar for x in self.data]
        return CSCMatrix(values, self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        from CSR import CSRMatrix
        """
        Транспонирование CSC матрицы.
        Hint:
        Результат - в CSR формате (с теми же данными, но с интерпретацией строк как столбцов).
        """

        return CSRMatrix(self.data, self.indices, self.indptr, (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        if self.shape[1] != other.shape[0]:
            raise ValueError("Умножение матриц невозможно")
        if not isinstance(other, CSCMatrix):
            if hasattr(other, "_to_csc"):
                other = other._to_csc()
            else:
                raise TypeError("Матрица должна быть CSC")

        res_csr = self._to_csr() @ other._to_csr()
        return res_csr._to_csc()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        rows_num = len(dense_matrix)
        cols_num = len(dense_matrix[0])
        data = []
        indices = []
        indptr = [0]

        for c in range(cols_num):
            for r in range(rows_num):
                value = dense_matrix[r][c]
                if value != 0:
                    data.append(value)
                    indices.append(r)
            indptr.append(len(data))

        return cls(data, indices, indptr, (rows_num, cols_num))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        coo = self._to_coo()
        return coo._to_csr()

    def _to_coo(self) -> 'COOMatrix':
        from COO import COOMatrix
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        rows = self.indices
        cols = []
        for col in range(self.shape[1]):
            count = self.indptr[col + 1] - self.indptr[col]
            cols.extend([col] * count)

        return COOMatrix(self.data, rows, cols, self.shape)
