from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        m, n = self.shape
        matrix = [[0 for _ in range(n)] for _ in range(m)]
        for row in range(m):
            start = self.indptr[row]
            end = self.indptr[row + 1]
            for idx in range(start, end):
                col = self.indices[idx]
                value = self.data[idx]
                if row < m and col < n:
                    matrix[row][col] = value
        return matrix

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        dense_self = self.to_dense()
        dense_other = other.to_dense()
        m, n = self.shape
        result_dense = [[dense_self[i][j] + dense_other[i][j] for j in range(n)] for i in range(m)]
        return self.from_dense(result_dense)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        new_data = [value * scalar for value in self.data]
        return CSRMatrix(new_data, self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Результат - в CSC формате (с теми же данными, но с интерпретацией столбцов как строк).
        """
        from CSC import CSCMatrix
        m, n = self.shape
        new_shape = (n, m)
        m_new, n_new = new_shape
        new_indptr = [0] * (n_new + 1)
        for col_idx in self.indices:
            if col_idx < n_new:
                new_indptr[col_idx + 1] += 1
        for i in range(1, n_new + 1):
            new_indptr[i] += new_indptr[i - 1]
        new_indices = [0] * len(self.indices)
        new_data = [0] * len(self.data)
        current_pos = new_indptr[:]
        for row in range(len(self.indptr) - 1):
            start = self.indptr[row]
            end = self.indptr[row + 1]
            for idx in range(start, end):
                col = self.indices[idx]
                if col < n_new:
                    pos = current_pos[col]
                    new_indices[pos] = row
                    new_data[pos] = self.data[idx]
                    current_pos[col] += 1
        return CSCMatrix(new_data, new_indices, new_indptr, new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        m, n = self.shape
        n2, p = other.shape
        dense_self = self.to_dense()
        dense_other = other.to_dense()
        result = [[0 for _ in range(p)] for _ in range(m)]
        for i in range(m):
            for j in range(p):
                for k in range(n):
                    result[i][j] += dense_self[i][k] * dense_other[k][j]
        return self.from_dense(result)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        if not dense_matrix:
            return cls([], [], [0], (0, 0))
        m = len(dense_matrix)
        n = len(dense_matrix[0]) if m > 0 else 0
        indptr = [0] * (m + 1)
        row_counts = [0] * m
        for i in range(m):
            for j in range(n):
                if dense_matrix[i][j] != 0:
                    row_counts[i] += 1
        for i in range(m):
            indptr[i + 1] = indptr[i] + row_counts[i]
        temp_data = [0] * (indptr[m])
        temp_indices = [0] * (indptr[m])
        current_pos = indptr[:]
        for i in range(m):
            for j in range(n):
                val = dense_matrix[i][j]
                if val != 0:
                    pos = current_pos[i]
                    temp_data[pos] = val
                    temp_indices[pos] = j
                    current_pos[i] += 1
        return cls(temp_data, temp_indices, indptr, (m, n))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSRMatrix в CSCMatrix.
        """
        return self.transpose()

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSRMatrix в COOMatrix.
        """
        from COO import COOMatrix
        m, n = self.shape
        data = []
        rows = []
        cols = []
        for row in range(m):
            start = self.indptr[row]
            end = self.indptr[row + 1]
            for idx in range(start, end):
                col = self.indices[idx]
                value = self.data[idx]
                data.append(value)
                rows.append(row)
                cols.append(col)
        return COOMatrix(data, rows, cols, self.shape)