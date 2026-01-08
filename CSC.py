from base import Matrix
from matrix_types import *
from COO import COOMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу."""
        n, m = self.shape
        result = [[0 for _ in range(m)] for _ in range(n)]

        for col in range(m):
            for i in range(self.indptr[col], self.indptr[col + 1]):
                row = self.indices[i]
                result[row][col] = self.data[i]

        return result

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        if self.shape != other.shape:
            raise ValueError("Размерности матриц не совпадают")

        n, m = self.shape
        result = [[0 for _ in range(m)] for _ in range(n)]

        for col in range(m):
            for i in range(self.indptr[col], self.indptr[col + 1]):
                row = self.indices[i]
                result[row][col] += self.data[i]

        for col in range(m):
            for i in range(other.indptr[col], other.indptr[col + 1]):
                row = other.indices[i]
                result[row][col] += other.data[i]

        data = []
        indices = []
        indptr = [0] * (m + 1)

        for col in range(m):
            for row in range(n):
                if result[row][col] != 0:
                    data.append(result[row][col])
                    indices.append(row)
            indptr[col + 1] = len(data)

        return CSCMatrix(data, indices, indptr, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        data = [self.data[i] * scalar for i in range(len(self.data))]
        return CSCMatrix(data, self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Hint:
        Результат - в CSR формате (с теми же данными, но с интерпретацией строк как столбцов).
        """
        from CSR import CSRMatrix

        return CSRMatrix(
            data=self.data,
            indices=self.indices,
            indptr=self.indptr,
            shape=(self.shape[1], self.shape[0])
        )
        
    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        res = self.to_dense() @ other.to_dense()
        return res.from_dense()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        data = []
        indices = []
        indptr = [0]

        matrix = dense_matrix
        n = len(matrix)
        m = len(matrix[0]) if n > 0 else 0

        for j in range(m):
            for i in range(n):
                if matrix[i][j] != 0:
                    data.append(matrix[i][j])
                    indices.append(i)
            indptr.append(len(data))

        return CSCMatrix(data, indices, indptr, (n, m))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix

        dense = self.to_dense()
        return CSRMatrix.from_dense(dense) 

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        dense = self.to_dense()
        return COOMatrix.from_dense(dense) 
