from base import Matrix
from mtypes import COOData, COORows, COOCols, Shape, DenseMatrix


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)

        self.data = data
        self.row = row
        self.col = col

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]

        for v, i, j in zip(self.data, self.row, self.col):
            dense[i][j] = v

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        a = self.to_dense()
        b = other.to_dense()

        rows, cols = self.shape
        res = [[a[i][j] + b[i][j] for j in range(cols)] for i in range(rows)]

        return COOMatrix.from_dense(res)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        if scalar == 0:
            return COOMatrix([], [], [], self.shape)

        return COOMatrix(
            [v * scalar for v in self.data],
            self.row[:],
            self.col[:],
            self.shape
        )

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        r, c = self.shape
        return COOMatrix(
            self.data[:],
            self.col[:],
            self.row[:],
            (c, r)
        )

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""
        A = self.to_dense()
        B = other.to_dense()

        rows_a, cols_a = self.shape
        _, cols_b = other.shape

        res = [[0.0] * cols_b for _ in range(rows_a)]

        for i in range(rows_a):
            for j in range(cols_a):
                if A[i][j] != 0:
                    for k in range(cols_b):
                        if B[j][k] != 0:
                            res[i][k] += A[i][j] * B[j][k]

        return COOMatrix.from_dense(res)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows else 0

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
        return CSCMatrix.from_dense(self.to_dense())

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix
        return CSRMatrix.from_dense(self.to_dense())