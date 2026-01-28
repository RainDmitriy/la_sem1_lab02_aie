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
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]

        for col in range(cols):
            start = self.indptr[col]
            end = self.indptr[col + 1]
            for idx in range(start, end):
                dense[self.indices[idx]][col] = self.data[idx]

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        if self.shape != other.shape:
            raise ValueError("матрицы разного размера")

        A = self.to_dense()
        B = other.to_dense()

        rows, cols = self.shape
        res = [[A[i][j] + B[i][j] for j in range(cols)] for i in range(rows)]

        return CSCMatrix.from_dense(res)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        if scalar == 0:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)

        new_data = [val * scalar for val in self.data]

        return CSCMatrix(new_data, self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Hint:
        Результат - в CSR формате (с теми же данными, но с интерпретацией строк как столбцов).
        """
        from CSR import CSRMatrix
        rows, cols = self.shape
        nnz = len(self.data)

        indptr = [0] * (cols + 1)
        for row_idx in self.indices:
            indptr[row_idx + 1] += 1
        for i in range(1, cols + 1):
            indptr[i] += indptr[i - 1]

        data = [0.0] * nnz
        indices = [0] * nnz
        counter = indptr[:]

        for col in range(cols):
            for idx in range(self.indptr[col], self.indptr[col + 1]):
                row = self.indices[idx]
                val = self.data[idx]
                pos = counter[row]
                data[pos] = val
                indices[pos] = col
                counter[row] += 1

        return CSRMatrix(data, indices, indptr, (cols, rows))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        dense_self = self.to_dense()
        dense_other = other.to_dense()

        rows_a, cols_a = self.shape
        rows_b, cols_b = other.shape

        result = [[0.0] * cols_b for _ in range(rows_a)]

        for i in range(rows_a):
            for j in range(cols_a):
                if dense_self[i][j] != 0:
                    for k in range(cols_b):
                        if dense_other[j][k] != 0:
                            result[i][k] += dense_self[i][j] * dense_other[j][k]

        return CSCMatrix.from_dense(result)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0

        data = []
        indices = []
        indptr = [0]

        for j in range(cols):
            count = 0
            for i in range(rows):
                val = dense_matrix[i][j]
                if val != 0:
                    data.append(val)
                    indices.append(i)
                    count += 1
            indptr.append(indptr[-1] + count)

        return cls(data, indices, indptr, (rows, cols))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix
        return CSRMatrix.from_dense(self.to_dense())

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        from COO import COOMatrix
        return COOMatrix.from_dense(self.to_dense())