from base import Matrix
from types import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix
from COO import COOMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr

        if len(indptr) != shape[0] + 1:
            raise ValueError(f"indptr должен иметь длину shape[0] + 1 = {shape[0] + 1}")

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]

        for i in range(rows):
            start, end = self.indptr[i], self.indptr[i + 1]
            for pos in range(start, end):
                j = self.indices[pos]
                val = self.data[pos]
                dense[i][j] = val

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        dense_self = self.to_dense()
        dense_other = other.to_dense()

        rows, cols = self.shape
        result_dense = [[0.0] * cols for _ in range(rows)]

        for i in range(rows):
            for j in range(cols):
                result_dense[i][j] = dense_self[i][j] + dense_other[i][j]

        return CSRMatrix.from_dense(result_dense)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        new_data = [val * scalar for val in self.data]
        return CSRMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Результат - в CSC формате.
        """
        coo = self._to_coo()
        transposed_coo = coo.transpose()
        return transposed_coo._to_csc()

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
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

        return CSRMatrix.from_dense(result_dense)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0

        data = []
        indices = []
        indptr = [0]

        for i in range(rows):
            row_nnz = 0
            for j in range(cols):
                val = dense_matrix[i][j]
                if val != 0:
                    data.append(val)
                    indices.append(j)
                    row_nnz += 1
            indptr.append(indptr[-1] + row_nnz)

        return cls(data, indices, indptr, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSRMatrix в CSCMatrix.
        """
        # Преобразуем через COO
        coo = self._to_coo()
        return coo._to_csc()

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSRMatrix в COOMatrix.
        """
        rows, cols = self.shape
        data = []
        row_indices = []
        col_indices = []

        for i in range(rows):
            start, end = self.indptr[i], self.indptr[i + 1]
            for pos in range(start, end):
                data.append(self.data[pos])
                row_indices.append(i)
                col_indices.append(self.indices[pos])

        return COOMatrix(data, row_indices, col_indices, self.shape)