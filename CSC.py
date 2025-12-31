from base import Matrix
from types import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        pass

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу."""
        pass

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        pass

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        pass

    def transpose(self) -> 'Matrix':
        """Транспонирование CSC матрицы."""
        pass

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        pass

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        pass

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        pass

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        pass
