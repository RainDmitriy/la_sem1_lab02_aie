from base import Matrix
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = list(data)
        self.indices = list(indices)
        self.indptr = list(indptr)

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу."""
        n, m = self.shape
        dense_matrix = [[0 for _ in range(m)] for _ in range(n)]
        for j in range(m):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for idx in range(start, end):
                i = self.indices[idx]
                dense_matrix[i][j] += self.data[idx]
        return dense_matrix

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        A = self.to_dense()
        B = other.to_dense()
        n, m = self.shape
        C = [[A[i][j] + B[i][j] for j in range(m)] for i in range(n)]
        return CSCMatrix.from_dense(C)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        data = [d * scalar for d in self.data]
        return CSCMatrix(data, self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Hint:
        Результат - в CSR формате (с теми же данными, но с интерпретацией строк как столбцов).
        """
        from CSR import CSRMatrix
        n, m = self.shape
        return CSRMatrix(
            data=self.data,
            indices=self.indices,
            indptr=self.indptr,
            shape=(m, n)
        )

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        A = self.to_dense()
        B = other.to_dense()
        n, k = self.shape
        _, m = other.shape
        C = [[0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                for l in range(k):
                    C[i][j] += A[i][l] * B[l][j]
        return CSCMatrix.from_dense(C)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        data, indices, indptr = [], [], [0]
        n = len(dense_matrix)
        m = len(dense_matrix[0]) if n > 0 else 0
        for j in range(m):
            for i in range(n):
                if dense_matrix[i][j] != 0:
                    data.append(dense_matrix[i][j])
                    indices.append(i)
            indptr.append(len(data))
        return cls(data, indices, indptr, (n, m))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix
        n, m = self.shape
        return CSRMatrix(
            data=self.data,
            indices=self.indices,
            indptr=self.indptr,
            shape=(n, m)
        )

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        from COO import COOMatrix

        row, col, data = [], [], []
        n, m = self.shape
        for j in range(m):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for idx in range(start, end):
                row.append(self.indices[idx])
                col.append(j)
                data.append(self.data[idx])

        return COOMatrix(data, row, col, (n, m))
