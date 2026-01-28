from base import Matrix
from type1 import COOData, COORows, COOCols, Shape, DenseMatrix


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.row = row
        self.col = col

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу."""
        self.denseMatrix: DenseMatrix = []
        for i in range(self.shape[0]):
            self.denseMatrix.append([0] * self.shape[1])

        for i in range(len(self.data)):
            r = self.row[i]
            c = self.col[i]
            self.denseMatrix[r][c] = self.data[i]

        return self.denseMatrix

    def _add_impl(self, other: "Matrix") -> "Matrix":
        """Сложение COO матриц."""
        ans_dense: DenseMatrix = []
        for i in range(other.shape[0]):
            ans_dense.append([0] * self.shape[1])

        self_dense = self.to_dense()
        other_dense = other.to_dense()

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                ans_dense[i][j] = self_dense[i][j] + other_dense[i][j]

        return self.from_dense(ans_dense)

    def _mul_impl(self, scalar: float) -> "Matrix":
        """Умножение COO на скаляр."""
        new_data = [val * scalar for val in self.data]
        return COOMatrix(new_data, self.row, self.col, self.shape)

    def transpose(self) -> "Matrix":
        """Транспонирование COO матрицы."""
        ans = COOMatrix([], [], [], [self.shape[1], self.shape[0]])
        ans.col = self.row
        ans.row = self.col
        ans.data = self.data

        return ans

    def _matmul_impl(self, other: "Matrix") -> "Matrix":
        """Умножение COO матриц."""
        ans: DenseMatrix = []
        for i in range(other.shape[1]):
            ans.append([0] * self.shape[0])

        self.to_dense()
        other.to_dense()
        for m in range(self.shape[0]):
            for k in range(other.shape[1]):
                x = 0
                for n in range(self.shape[1]):
                    x += self.denseMatrix[m][n] * other.denseMatrix[n][k]
                ans[m][k] = x

        return self.from_dense(ans)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> "COOMatrix":
        """Создание COO из плотной матрицы."""
        row = []
        col = []
        data = []
        shape = [len(dense_matrix), len(dense_matrix[0])]
        for cur_row in range(len(dense_matrix)):
            for cur_col in range(len(dense_matrix[cur_row])):
                val = dense_matrix[cur_row][cur_col]
                if val != 0:
                    row.append(cur_row)
                    col.append(cur_col)
                    data.append(dense_matrix[cur_row][cur_col])
        return cls(data, row, col, shape)

    def _to_csc(self) -> "CSCMatrix":
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix

        indptr = [0] * (self.shape[1] + 1)
        for col in self.col:
            indptr[col + 1] += 1
        for j in range(self.shape[1]):
            indptr[j + 1] += indptr[j]
        return CSCMatrix(self.data, self.row, indptr, self.shape)

    def _to_csr(self) -> "CSRMatrix":
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix

        indptr = [0] * (self.shape[0] + 1)
        for row in self.row:
            indptr[row + 1] += 1
        for i in range(self.shape[0]):
            indptr[i + 1] += indptr[i]
        return CSRMatrix(self.data, self.col, indptr, self.shape)
