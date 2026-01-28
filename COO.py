from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.row = row
        self.col = col

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу."""
        m, n = self.shape
        matrix = [[0 for _ in range(n)] for _ in range(m)]
        if not self.data:
            return matrix
        i = 0
        for r, c in zip(self.row, self.col):
            if r >= m or c >= n:
                i += 1
                continue
            matrix[r][c] = self.data[i]
            i += 1
        return matrix

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц."""
        dense_self = self.to_dense()
        dense_other = other.to_dense()
        m, n = self.shape
        data = []
        row = []
        col = []
        for i in range(m):
            for j in range(n):
                value = dense_self[i][j] + dense_other[i][j]
                if value != 0:
                    data.append(value)
                    row.append(i)
                    col.append(j)
        return COOMatrix(data, row, col, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        for i in range(len(self.data)):
            self.data[i] *= scalar
        return COOMatrix(self.data, self.row, self.col, self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        new_shape = (self.shape[1], self.shape[0])
        return COOMatrix(self.data, self.col, self.row, new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        m, n = self.shape
        n2, p = other.shape
        dense_self = self.to_dense()
        dense_other = other.to_dense()
        result = [[0 for _ in range(p)] for _ in range(m)]
        for i in range(m):
            for j in range(p):
                for k in range(n):
                    result[i][j] += dense_self[i][k] * dense_other[k][j]
        return COOMatrix.from_dense(result)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        data = []
        row = []
        col = []
        m = len(dense_matrix)
        n = len(dense_matrix[0]) if m > 0 else 0
        for i in range(m):
            for j in range(n):
                val = dense_matrix[i][j]
                if val != 0:
                    data.append(val)
                    row.append(i)
                    col.append(j)
        return cls(data, row, col, (m, n))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix
        m, n = self.shape
        indptr = [0] * (n + 1)
        col_counts = [0] * n
        for c in self.col:
            col_counts[c] += 1
        for j in range(n):
            indptr[j + 1] = indptr[j] + col_counts[j]
        temp_data = [0] * len(self.data)
        temp_indices = [0] * len(self.data)
        current_pos = indptr[:]
        for idx in range(len(self.data)):
            c = self.col[idx]
            pos = current_pos[c]
            temp_data[pos] = self.data[idx]
            temp_indices[pos] = self.row[idx]
            current_pos[c] += 1
        data = temp_data[:]
        indices = temp_indices[:]
        return CSCMatrix(data, indices, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix
        m, n = self.shape
        indptr = [0] * (m + 1)
        row_counts = [0] * m
        for r in self.row:
            row_counts[r] += 1
        for i in range(m):
            indptr[i + 1] = indptr[i] + row_counts[i]
        temp_data = [0] * len(self.data)
        temp_indices = [0] * len(self.data)
        current_pos = indptr[:]
        for idx in range(len(self.data)):
            r = self.row[idx]
            pos = current_pos[r]
            temp_data[pos] = self.data[idx]
            temp_indices[pos] = self.col[idx]
            current_pos[r] += 1
        data = temp_data[:]
        indices = temp_indices[:]
        return CSRMatrix(data, indices, indptr, self.shape)
