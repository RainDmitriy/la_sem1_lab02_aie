from base import Matrix
from matrix_types import *
from COO import COOMatrix

class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        n, m = self.shape
        result = [[0 for _ in range(m)] for _ in range(n)]

        for row in range(n):
            for i in range(self.indptr[row], self.indptr[row + 1]):
                col = self.indices[i]
                result[row][col] = self.data[i]

        return result

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        if self.shape != other.shape:
            raise ValueError("Размерности матриц не совпадают")
        
        n, m = self.shape
        result = [[0 for _ in range(m)] for _ in range(n)]

        for i in range(n):
            for k in range(self.indptr[i], self.indptr[i + 1]):
                col = self.indices[k]
                result[i][col] += self.data[k]

        for i in range(n):
            for k in range(other.indptr[i], other.indptr[i + 1]):
                col = other.indices[k]
                result[i][col] += other.data[k]

        data = []
        indices = []
        indptr = [0]

        for i in range(n):
            for j in range(m):
                if result[i][j] != 0:
                    data.append(result[i][j])
                    indices.append(j)
            indptr.append(len(data))

        return CSRMatrix(data, indices, indptr, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        data = [self.data[i] * scalar for i in range(len(self.data))]
        return CSRMatrix(data, self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Hint:
        Результат - в CSC формате (с теми же данными, но с интерпретацией столбцов как строк).
        """
        from CSC import CSCMatrix

        return CSCMatrix(
            data=self.data,
            indices=self.indices,
            indptr=self.indptr,
            shape=(self.shape[1], self.shape[0])
        )

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        res = self.to_dense() @ other.to_dense()
        return res.from_dense()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        matrix = dense_matrix
        n = len(matrix)
        m = len(matrix[0]) if n > 0 else 0

        data = []
        indices = []
        indptr = [0]

        for i in range(n):
            for j in range(m):
                if matrix[i][j] != 0:
                    data.append(matrix[i][j])
                    indices.append(j)
            indptr.append(len(data))

        return cls(
            data=data,
            indices=indices,
            indptr=indptr,
            shape=(n, m)
        )

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSRMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix

        dense = self.to_dense()
        return CSCMatrix.from_dense(dense)
    
    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSRMatrix в COOMatrix.
        """
        dense = self.to_dense()
        return COOMatrix.from_dense(dense)