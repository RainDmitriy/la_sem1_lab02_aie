from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.data = list(data)
        self.row = list(row)
        self.col = list(col)

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу."""
        n, m = self.shape
        dense_matrix = [[0 for _ in range(m)] for _ in range(n)]
        for r, c, d in zip(self.row, self.col, self.data):
            dense_matrix[r][c] += d
        return dense_matrix

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц."""
        A = self.to_dense()
        B = other.to_dense()
        n, m = self.shape
        C = [[A[i][j] + B[i][j] for j in range(m)] for i in range(n)]
        return COOMatrix.from_dense(C)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        data = [d * scalar for d in self.data]
        return COOMatrix(data, self.row, self.col, self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        n, m = self.shape
        return COOMatrix(self.data, self.col, self.row, (m, n))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""
        A = self.to_dense()
        B = other.to_dense()
        n, k = self.shape
        _, m = other.shape
        C = [[0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                for l in range(k):
                    C[i][j] += A[i][l] * B[l][j]
        return COOMatrix.from_dense(C)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        data, row, col = [], [], []
        n = len(dense_matrix)
        if n > 0:
            m = len(dense_matrix[0])
        else:
            m = 0
        for i in range(n):
            for j in range(m):
                if dense_matrix[i][j] != 0:
                    data.append(dense_matrix[i][j])
                    row.append(i)
                    col.append(j)
        return cls(data, row, col, (n, m))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix

        n, m = self.shape
        data = []
        indices = []
        indptr = [0] * (m + 1)
        for c in self.col:
            indptr[c + 1] += 1
        for j in range(m):
            indptr[j + 1] += indptr[j]
        counter = indptr.copy()
        for r, c, v in zip(self.row, self.col, self.data):
            pos = counter[c]
            data.insert(pos, v)
            indices.insert(pos, r)
            counter[c] += 1

        return CSCMatrix(data, indices, indptr, (n, m))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix

        n, m = self.shape
        data = []
        indices = []
        indptr = [0] * (n + 1)
        for r in self.row:
            indptr[r + 1] += 1
        for i in range(n):
            indptr[i + 1] += indptr[i]
        counter = indptr.copy()
        for r, c, v in zip(self.row, self.col, self.data):
            pos = counter[r]
            data.insert(pos, v)
            indices.insert(pos, c)
            counter[r] += 1

        return CSRMatrix(data, indices, indptr, (n, m))
