from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = list(data)
        self.indices = list(indices)
        self.indptr = list(indptr)

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        n, m = self.shape
        dense_matrix = [[0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for idx in range(start, end):
                j = self.indices[idx]
                dense_matrix[i][j] += self.data[idx]
        return dense_matrix

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        A = self.to_dense()
        B = other.to_dense()
        n, m = self.shape
        C = [[A[i][j] + B[i][j] for j in range(m)] for i in range(n)]
        return CSRMatrix.from_dense(C)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        data = [d * scalar for d in self.data]
        return CSRMatrix(data, self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Hint:
        Результат - в CSC формате (с теми же данными, но с интерпретацией столбцов как строк).
        """
        from CSC import CSCMatrix
        n, m = self.shape
        return CSCMatrix(
            data=self.data,
            indices=self.indices,
            indptr=self.indptr,
            shape=(m, n)
        )

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        A = self.to_dense()
        B = other.to_dense()
        n, k = self.shape
        _, m = other.shape
        C = [[0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                for l in range(k):
                    C[i][j] += A[i][l] * B[l][j]
        return CSRMatrix.from_dense(C)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        data, indices, indptr = [], [], [0]
        n = len(dense_matrix)
        m = len(dense_matrix[0]) if n > 0 else 0
        for i in range(n):
            for j in range(m):
                if dense_matrix[i][j] != 0:
                    data.append(dense_matrix[i][j])
                    indices.append(j)
            indptr.append(len(data))
        return cls(data, indices, indptr, (n, m))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSRMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix

        n, m = self.shape
        col_counts = [0] * m
        for col in self.indices:
            col_counts[col] += 1
        indptr = [0] * (m + 1)
        for j in range(m):
            indptr[j + 1] = indptr[j] + col_counts[j]
        counter = indptr.copy()
        data = [0] * len(self.data)
        indices = [0] * len(self.data)
        for i in range(n):
            for pos in range(self.indptr[i], self.indptr[i + 1]):
                j = self.indices[pos]
                idx = counter[j]
                data[idx] = self.data[pos]
                indices[idx] = i
                counter[j] += 1

        return CSCMatrix(data, indices, indptr, (n, m))

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSRMatrix в COOMatrix.
        """
        from COO import COOMatrix

        row, col, data = [], [], []
        n, _ = self.shape
        for i in range(n):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for idx in range(start, end):
                row.append(i)
                col.append(self.indices[idx])
                data.append(self.data[idx])

        return COOMatrix(data, row, col, self.shape)
