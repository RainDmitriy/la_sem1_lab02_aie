from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix
from collections import defaultdict

class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]

        for row in range(rows):
            start = self.indptr[row]
            end = self.indptr[row + 1]
            for idx in range(start, end):
                col = self.indices[idx]
                dense[row][col] = self.data[idx]

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        return self._to_coo()._add_impl(other._to_coo())._to_csr()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        if scalar == 0:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)

        new_data = [v * scalar for v in self.data]
        return CSRMatrix(new_data, self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Hint:
        Результат - в CSC формате (с теми же данными, но с интерпретацией столбцов как строк).
        """
        return self._to_coo()._to_csc().transpose()

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        return self._to_coo()._matmul_impl(other._to_coo())._to_csr()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        from COO import COOMatrix
        return COOMatrix.from_dense(dense_matrix)._to_csr()

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSRMatrix в CSCMatrix.
        """
        return self._to_coo()._to_csc()

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSRMatrix в COOMatrix.
        """
        from COO import COOMatrix
        data, row, col = [], [], []
        rows = self.shape[0]
        for i in range(rows):
            for idx in range(self.indptr[i], self.indptr[i + 1]):
                row.append(i)
                col.append(self.indices[idx])
                data.append(self.data[idx])
        return COOMatrix(data, row, col, self.shape)
