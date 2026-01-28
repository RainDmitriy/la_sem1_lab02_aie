from base import Matrix
from type1 import COOData, COORows, COOCols, Shape, DenseMatrix


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        self.shape = shape
        self.data = data
        self.row = row
        self.col = col

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу."""
        matrix: DenseMatrix = [[0] * self.shape[1] for i in range(self.shape[0])]
        for i in range(len(self.data)):
            matrix[self.row[i]][self.col[i]] = self.data[i]
        return matrix

    def _add_impl(self, other: "Matrix") -> "Matrix":
        """Сложение COO матриц."""
        coords = {}

        for i in range(len(self.data)):
            pos = (self.row[i], self.col[i])
            coords[pos] = self.data[i]

        for i in range(len(other.data)):
            pos = (other.row[i], other.col[i])
            coords[pos] = coords.get(pos, 0) + other.data[i]

        ans_data, ans_row, ans_col = [], [], []
        for (r, c), val in coords.items():
            if val != 0:
                ans_data.append(val)
                ans_row.append(r)
                ans_col.append(c)

        return COOMatrix(ans_data, ans_row, ans_col, self.shape)

    def _mul_impl(self, scalar: float) -> "Matrix":
        """Умножение COO на скаляр."""
        for i in range(len(self.data)):
            self.data[i] *= scalar
        return self

    def transpose(self) -> "Matrix":
        """Транспонирование COO матрицы."""
        self.row, self.col = self.col, self.row
        return self

    def _matmul_impl(self, other: "Matrix") -> "Matrix":
        """Умножение COO матриц."""
        ans: DenseMatrix = []
        for i in range(self.shape[0]):
            ans.append([0] * other.shape[1])
        self_dense = self.to_dense()
        other_dense = other.to_dense()

        for m in range(self.shape[0]):
            for k in range(other.shape[1]):
                x = 0
                for n in range(self.shape[1]):
                    x += self_dense[m][n] * other_dense[n][k]
                ans[m][k] = x

        return COOMatrix.from_dense(ans)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> "COOMatrix":
        """Создание COO из плотной матрицы."""
        data, row, col = [], [], []
        shape = (len(dense_matrix), len(dense_matrix[0]))

        for r in range(shape[0]):
            for c in range(shape[1]):
                if dense_matrix[r][c] != 0:
                    data.append(dense_matrix[r][c])
                    row.append(r)
                    col.append(c)
        return cls(data, row, col, shape)

    def _to_csc(self) -> "CSCMatrix":
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix

        items = sorted(zip(self.row, self.col, self.data), key=lambda x: (x[1], x[0]))

        new_data = []
        new_indices = []
        indptr = [0] * (self.shape[1] + 1)

        for r, c, v in items:
            new_data.append(v)
            new_indices.append(r)
            indptr[c + 1] += 1

        for j in range(self.shape[1]):
            indptr[j + 1] += indptr[j]

        return CSCMatrix(new_data, new_indices, indptr, self.shape)

    def _to_csr(self) -> "CSRMatrix":
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix

        items = sorted(zip(self.row, self.col, self.data))

        new_data = []
        new_indices = []
        indptr = [0] * (self.shape[0] + 1)

        for r, c, v in items:
            new_data.append(v)
            new_indices.append(c)
            indptr[r + 1] += 1

        for i in range(self.shape[0]):
            indptr[i + 1] += indptr[i]

        return CSRMatrix(new_data, new_indices, indptr, self.shape)

