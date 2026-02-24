from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        if not (len(data) == len(row) == len(col)):
            raise ValueError("COO: data, row и col должны быть одинаковой длины")
        self.data = list(data)
        self.row = list(row)
        self.col = list(col)

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0 for _ in range(cols)] for _ in range(rows)]
        for value, row, col in zip(self.data, self.row, self.col):
            dense[row][col] += value
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц."""
        if not isinstance(other, COOMatrix):
            return COOMatrix.from_dense(
                [[a + b for a, b in zip(row_a, row_b)]
                 for row_a, row_b in zip(self.to_dense(), other.to_dense())]
            )

        result = {}

        for data, row, col in zip(self.data, self.row, self.col):
            result[(row, col)] = result.get((row, col), 0) + data

        for data, row, col in zip(other.data, other.row, other.col):
            result[(row, col)] = result.get((row, col), 0) + data

        data, row, col = [], [], []
        for (row, col), data in result.items():
            if data != 0:
                data.append(data)
                row.append(row)
                col.append(col)

        return COOMatrix(data, row, col, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        if scalar == 0:
            return COOMatrix([], [], [], self.shape)

        data = [dat * scalar for dat in self.data]
        return COOMatrix(data, self.row, self.col, self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        return COOMatrix(
            self.data.copy(),
            self.col.copy(),
            self.row.copy(),
            (self.shape[1], self.shape[0])
        )

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""
        A = self.to_dense()
        B = other.to_dense()

        n, m = self.shape
        _, p = other.shape

        result = [[0 for _ in range(p)] for _ in range(n)]

        for i in range(n):
            for k in range(m):
                if A[i][k] != 0:
                    for j in range(p):
                        if B[k][j] != 0:
                            result[i][j] += A[i][k] * B[k][j]

        return COOMatrix.from_dense(result)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0

        data, row, col = [], [], []

        for i in range(rows):
            for j in range(cols):
                if dense_matrix[i][j] != 0:
                    data.append(dense_matrix[i][j])
                    row.append(i)
                    col.append(j)

        return cls(data, row, col, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix

        rows, cols = self.shape
        nnz = len(self.data)

        col_ptr = [0] * (cols + 1)
        for col in self.col:
            col_ptr[col + 1] += 1

        for i in range(1, len(col_ptr)):
            col_ptr[i] += col_ptr[i - 1]

        data = [0] * nnz
        row_ind = [0] * nnz
        counter = col_ptr.copy()

        for data, row, col in zip(self.data, self.row, self.col):
            idx = counter[col]
            data[idx] = data
            row_ind[idx] = row
            counter[col] += 1

        return CSCMatrix(data, row_ind, col_ptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix

        rows, cols = self.shape
        nnz = len(self.data)

        row_ptr = [0] * (rows + 1)
        for r in self.row:
            row_ptr[r + 1] += 1

        for i in range(1, len(row_ptr)):
            row_ptr[i] += row_ptr[i - 1]

        data = [0] * nnz
        col_ind = [0] * nnz
        counter = row_ptr.copy()

        for data, row, col in zip(self.data, self.row, self.col):
            idx = counter[row]
            data[idx] = data
            col_ind[idx] = col
            counter[row] += 1

        return CSRMatrix(data, col_ind, row_ptr, self.shape)
