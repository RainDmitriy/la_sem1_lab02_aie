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

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц."""
        for i in range(len(other.data)):
            for j in range(len(self.data)):
                if (other.row[i] == self.row[j] and other.col[i] == self.col[j]):
                    self.data[i] += other.data[j]
                    break
            else:
                self.data.append(other.data[i])
                self.col.append(other.col[i])
                self.row.append(other.row[i])
        return self

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        for i in range(len(self.data)):
            self.data[i] *= scalar
        return self

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        self.row, self.col = self.col, self.row
        return self

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
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
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
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

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix
        data, indices, indptr = [], [], [0]
        for i in range(self.shape[1]):
            count = 0
            for j in range(len(self.col)):
                if i == self.col[j]:
                    count += 1
                    data.append(self.data[j])
                    indices.append(self.row[j])
            indptr.append(indptr[-1] + count)

        return CSCMatrix(data, indices, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix
        data, indices, indptr = [], [], [0]
        for i in range(self.shape[0]):
            count = 0
            for j in range(len(self.row)):
                if i == self.row[j]:
                    count += 1
                    data.append(self.data[j])
                    indices.append(self.col[j])
            indptr.append(indptr[-1] + count)

        return CSRMatrix(data, indices, indptr, self.shape)

