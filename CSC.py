from base import Matrix
from types import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix
from COO import COOMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr

        if len(indptr) != shape[1] + 1:
            raise ValueError(f"indptr должен иметь длину shape[1] + 1 = {shape[1] + 1}")

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]

        for j in range(cols):
            start, end = self.indptr[j], self.indptr[j + 1]
            for pos in range(start, end):
                i = self.indices[pos]
                val = self.data[pos]
                dense[i][j] = val

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        dense_self = self.to_dense()
        dense_other = other.to_dense()
        rows, cols = self.shape
        result_dense = [[0.0] * cols for _ in range(rows)]

        for i in range(rows):
            for j in range(cols):
                result_dense[i][j] = dense_self[i][j] + dense_other[i][j]

        return CSCMatrix.from_dense(result_dense)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        new_data = [val * scalar for val in self.data]
        return CSCMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Результат - в CSR формате.
        """
        # Для транспонирования преобразуем через COO
        coo = self._to_coo()
        transposed_coo = coo.transpose()
        return transposed_coo._to_csr()

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        dense_self = self.to_dense()
        dense_other = other.to_dense()
        rows, cols = self.shape[0], other.shape[1]
        inner = self.shape[1]
        result_dense = [[0.0] * cols for _ in range(rows)]

        for i in range(rows):
            for j in range(cols):
                total = 0.0
                for k in range(inner):
                    total += dense_self[i][k] * dense_other[k][j]
                result_dense[i][j] = total

        return CSCMatrix.from_dense(result_dense)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        # в coo
        coo = COOMatrix.from_dense(dense_matrix)
        # в csc
        return coo._to_csc()

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        coo = self._to_coo()
        return coo._to_csr()

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        rows, cols = self.shape
        data = []
        row_indices = []
        col_indices = []

        for j in range(cols):
            start, end = self.indptr[j], self.indptr[j + 1]
            for pos in range(start, end):
                data.append(self.data[pos])
                row_indices.append(self.indices[pos])
                col_indices.append(j)

        return COOMatrix(data, row_indices, col_indices, self.shape)