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
        h, w = self.shape
        res = [[0.0] * w for _ in range(h)]
        for r, c, v in zip(self.row, self.col, self.data):
            res[r][c] += v
        return res

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц."""
        return COOMatrix(
            self.data + other.data,
            self.row + other.row,
            self.col + other.col,
            self.shape
        )

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        new_d = [x * scalar for x in self.data]
        return COOMatrix(new_d, list(self.row), list(self.col), self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        return COOMatrix(
            list(self.data),
            list(self.col),
            list(self.row),
            (self.shape[1], self.shape[0])
        )

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""
        if self.shape[1] != other.shape[0]:
            raise ValueError("Size mismatch")

        tmp = {}
        b_map = {}

        for r, c, v in zip(other.row, other.col, other.data):
            if r not in b_map: b_map[r] = []
            b_map[r].append((c, v))

        for r, c, v in zip(self.row, self.col, self.data):
            if c in b_map:
                for cb, vb in b_map[c]:
                    key = (r, cb)
                    tmp[key] = tmp.get(key, 0.0) + v * vb

        d, r, c = [], [], []
        for (row, col), val in tmp.items():
            r.append(row)
            c.append(col)
            d.append(val)

        return COOMatrix(d, r, c, (self.shape[0], other.shape[1]))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        h = len(dense_matrix)
        w = len(dense_matrix[0]) if h else 0
        d, r, c = [], [], []

        for i in range(h):
            for j in range(w):
                val = dense_matrix[i][j]
                if val != 0:
                    d.append(val)
                    r.append(i)
                    c.append(j)

        return cls(d, r, c, (h, w))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix

        s = sorted(zip(self.row, self.col, self.data), key=lambda x: (x[1], x[0]))
        d = [x[2] for x in s]
        ind = [x[0] for x in s]
        ptr = [0] * (self.shape[1] + 1)

        for _, c, _ in s:
            ptr[c + 1] += 1

        for i in range(self.shape[1]):
            ptr[i + 1] += ptr[i]

        return CSCMatrix(d, ind, ptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix

        s = sorted(zip(self.row, self.col, self.data))
        d = [x[2] for x in s]
        ind = [x[1] for x in s]
        ptr = [0] * (self.shape[0] + 1)

        for r, _, _ in s:
            ptr[r + 1] += 1

        for i in range(self.shape[0]):
            ptr[i + 1] += ptr[i]

        return CSRMatrix(d, ind, ptr, self.shape)