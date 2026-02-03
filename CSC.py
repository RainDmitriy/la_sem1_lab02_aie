from base import Matrix
from types import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу."""
        dense = [[0.0 for _ in range(self.shape[1])] for _ in range(self.shape[0])]
        for j in range(self.shape[1]):
            for i in range(self.indptr[j], self.indptr[j + 1]):
                dense[self.indices[i]][j] = self.data[i]
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        return self._to_coo()._add_impl(other._to_coo())._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        return CSCMatrix([v * scalar for v in self.data], self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Hint:
        Результат - в CSR формате (с теми же данными, но с интерпретацией строк как столбцов).
        """
        from CSR import CSRMatrix
        return CSRMatrix(self.data[:], self.indices[:], self.indptr[:], (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        return self._to_csr()._matmul_impl(other)._to_csc()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        from COO import COOMatrix
        return COOMatrix.from_dense(dense_matrix)._to_csc()

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        return self._to_coo()._to_csr()

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        from COO import COOMatrix
        cols = []
        for j in range(self.shape[1]):
            for i in range(self.indptr[j], self.indptr[j + 1]):
                cols.append(j)
        return COOMatrix(self.data[:], self.indices[:], cols, self.shape)

    def _to_csc(self) -> 'CSCMatrix':
        return self
