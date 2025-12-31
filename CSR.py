from base import Matrix
from types import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        pass

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        pass

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        pass

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        pass

    def transpose(self) -> 'Matrix':
        """Транспонирование CSR матрицы."""
        pass

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        pass

    def _to_coo(self) -> 'COOMatrix':
        """Преобразование CSR в COO."""
        pass

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        pass

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSRMatrix в CSCMatrix.
        """
        pass
