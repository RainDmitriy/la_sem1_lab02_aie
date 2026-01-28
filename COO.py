from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix

TOLERANCE = 1e-8

class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        if not(len(data) == len(row) == len(col)):
            raise ValueError("data, row, col doesnt have same length")
        self.data = []
        self.row = []
        self.col = []
        for d, r, c in zip(data, row, col):
            if d != 0.0:
                self.data.append(d)
                self.row.append(r)
                self.col.append(c)

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу."""
        rows, cols = self.shape
        matrix = []
        for i in range(rows):
            row = []
            for j in range(cols):
                row.append(0.0)
            matrix.append(row)

        for i in range(len(self.data)):
            matrix[self.row[i]][self.col[i]] = self.data[i]
        return matrix

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц."""
        dense_self = self.to_dense()
        dense_other = other.to_dense()
        rows, cols = self.shape
        result_dense = []
        for i in range(rows):
            new_row = []
            for j in range(cols):
                sum_val = dense_self[i][j] + dense_other[i][j]
                new_row.append(sum_val)
            result_dense.append(new_row)

        return COOMatrix.from_dense(result_dense)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        if scalar == 0.0:
            return COOMatrix([], [], [], self.shape)
        new_data = [x * scalar for x in self.data]
        return COOMatrix(new_data, self.row[:], self.col[:], self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        new_data = []
        new_row = []
        new_col = []
        for i in range(len(self.data)):
            new_data.append(self.data[i])
            new_row.append(self.col[i])
            new_col.append(self.row[i])
        new_shape = (self.shape[1], self.shape[0])
        return COOMatrix(new_data, new_row, new_col, new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""
        if not isinstance(other, COOMatrix):
            B = other._to_coo()
        else:
            B = other
        rows_A, cols_A = self.shape
        rows_B, cols_B = other.shape
        if cols_A != rows_B:
            raise ValueError("matrix self' col and matrix other' row doesnt have same length")
        B_cols = [{} for _ in range(cols_B)]
        for val, r, c in zip(B.data, B.row, B.col):
            B_cols[c][r] = val
        A_rows = [[] for _ in range(rows_A)]
        for val, r, c in zip(self.data, self.row, self.col):
            A_rows[r].append((c, val))
        result_data = []
        result_row = []
        result_col = []
        for i in range(rows_A):
            for j in range(cols_B):
                total = 0.0
                for k, a_ik in A_rows[i]:
                    if k in B_cols[j]:
                        total += a_ik * B_cols[j][k]
                if abs(total) > TOLERANCE:
                    result_data.append(total)
                    result_row.append(i)
                    result_col.append(j)

        return COOMatrix(result_data, result_row, result_col, (rows_A, cols_B))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        if not dense_matrix:
            return cls([], [], [], (0, 0))
        rows = len(dense_matrix)
        if rows == 0:
            cols = 0
        else:
            cols = len(dense_matrix[0])
        data, row, col = [], [], []

        for i in range(rows):
            for j in range(cols):
                val = dense_matrix[i][j]
                if val != 0.0:
                    data.append(val)
                    row.append(i)
                    col.append(j)
        return cls(data, row, col, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix
        if not self.data:
            rows, cols = self.shape
            return CSCMatrix([], [], [0] * (cols + 1), (rows, cols))
        triples = sorted(zip(self.col, self.row, self.data))
        data = []
        indices = []
        indptr = [0]

        current_col = 0
        for col, row, val in triples:
            while current_col < col:
                indptr.append(len(data))
                current_col += 1
            data.append(val)
            indices.append(row)
        rows, cols = self.shape
        while current_col < cols:
            indptr.append(len(data))
            current_col += 1

        return CSCMatrix(data, indices, indptr, (rows, cols))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix
        if not self.data:
            rows, cols = self.shape
            return CSRMatrix([], [], [0] * (rows + 1), (rows, cols))
        triples = sorted(zip(self.row, self.col, self.data))
        data = []
        indices = []
        indptr = [0]

        current_row = 0
        for row, col, val in triples:
            while current_row < row:
                indptr.append(len(data))
                current_row += 1
            data.append(val)
            indices.append(col)
        rows, cols = self.shape
        while current_row < rows:
            indptr.append(len(data))
            current_row += 1

        return CSRMatrix(data, indices, indptr, (rows, cols))
