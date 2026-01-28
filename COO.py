from base import Matrix
from types import COOData, COORows, COOCols, Shape, DenseMatrix

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
        data = []
        row = []
        col = []

        temp = {}
        for v, r, c in zip(self.data, self.row, self.col):
            temp[(r, c)] = v
        for v, r, c in zip(other.data, other.row, other.col):
            temp[(r, c)] = temp.get((r, c), 0) + v

        for (r, c), v in temp.items():
            if v != 0:
                data.append(v)
                row.append(r)
                col.append(c)

        return COOMatrix(data, row, col, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        if scalar == 0:
            return COOMatrix([], [], [], self.shape)
        new_data = [v * scalar for v in self.data]
        return COOMatrix(new_data, self.row[:], self.col[:], self.shape)

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
        result = {}
        for v1, r1, c1 in zip(self.data, self.row, self.col):
            for v2, r2, c2 in zip(other.data, other.row, other.col):
                if c1 == r2:
                    result[(r1, c2)] = result.get((r1, c2), 0) + v1 * v2

        data = []
        row = []
        col = []
        for (r, c), v in result.items():
            if v != 0:
                data.append(v)
                row.append(r)
                col.append(c)

        shape = (self.shape[0], other.shape[1])
        return COOMatrix(data, row, col, shape)

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