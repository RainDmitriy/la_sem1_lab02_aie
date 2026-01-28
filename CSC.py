from base import Matrix
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data.copy()
        self.indices = indices.copy()
        self.indptr = indptr.copy()

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу."""
        m, n = self.shape
        matrix = [[0 for _ in range(n)] for _ in range(m)]
        for col in range(n):
            start = self.indptr[col]
            end = self.indptr[col + 1]
            for idx in range(start, end):
                row = self.indices[idx]
                value = self.data[idx]
                if row < m:
                    matrix[row][col] = value
        return matrix

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        dense_self = self.to_dense()
        dense_other = other.to_dense()
        m, n = self.shape
        result_dense = [[dense_self[i][j] + dense_other[i][j] for j in range(n)] for i in range(m)]
        return self.from_dense(result_dense)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        for i in range(len(self.data)):
            self.data[i] *= scalar
        return CSCMatrix(self.data, self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Hint:
        Результат - в CSR формате (с теми же данными, но с интерпретацией строк как столбцов).
        """
        from CSR import CSRMatrix
        m, n = self.shape
        new_shape = (n, m)
        m_new, n_new = new_shape
        new_indptr = [0] * (m_new + 1)
        for row_idx in self.indices:
            if row_idx < m_new:
                new_indptr[row_idx + 1] += 1
        for i in range(1, m_new + 1):
            new_indptr[i] += new_indptr[i - 1]
        new_indices = [0] * len(self.indices)
        new_data = [0] * len(self.data)
        current_pos = new_indptr[:]
        for col in range(len(self.indptr) - 1):
            start = self.indptr[col]
            end = self.indptr[col + 1]
            for idx in range(start, end):
                row = self.indices[idx]
                if row < m_new:
                    pos = current_pos[row]
                    new_indices[pos] = col
                    new_data[pos] = self.data[idx]
                    current_pos[row] += 1
        return CSRMatrix(new_data, new_indices, new_indptr, new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        m, n = self.shape
        n2, p = other.shape
        if n != n2:
            raise ValueError("Размеры матриц не совместимы для умножения")
        dense_other = other.to_dense()
        result = [[0.0] * p for _ in range(m)]
        for col in range(n):
            start = self.indptr[col]
            end = self.indptr[col + 1]
            for idx in range(start, end):
                row = self.indices[idx]
                val = self.data[idx]
                for j in range(p):
                    result[row][j] += val * dense_other[col][j]
        return self.from_dense(result)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        """Создание CSC из плотной матрицы."""
        if not dense_matrix or len(dense_matrix) == 0:
            return cls([], [], [0], (0, 0))
        m = len(dense_matrix)
        n = len(dense_matrix[0])
        data = []
        indices = []
        col_counts = [0] * n
        for col in range(n):
            for row in range(m):
                value = dense_matrix[row][col]
                if value != 0:
                    data.append(float(value))
                    indices.append(row)
                    col_counts[col] += 1
        indptr = [0] * (n + 1)
        for col in range(n):
            indptr[col + 1] = indptr[col] + col_counts[col]
        return cls(data, indices, indptr, (m, n))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        return self.transpose()

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        from COO import COOMatrix
        m, n = self.shape
        data = []
        rows = []
        cols = []
        for col in range(n):
            start = self.indptr[col]
            end = self.indptr[col + 1]
            for idx in range(start, end):
                row = self.indices[idx]
                value = self.data[idx]
                data.append(value)
                rows.append(row)
                cols.append(col)
        return COOMatrix(data, rows, cols, self.shape)
