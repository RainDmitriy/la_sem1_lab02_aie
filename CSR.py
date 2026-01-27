from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix, TOLERANCE

class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        n_rows, n_cols = self.shape
        dense = [[0.0 for _ in range(n_cols)] for _ in range(n_rows)]
        for i in range(n_rows):
            for idx in range(self.indptr[i], self.indptr[i + 1]):
                col_idx = self.indices[idx]
                dense[i][col_idx] = self.data[idx]
        return dense

    def _add_impl(self, other: 'CSRMatrix') -> 'CSRMatrix':
        """Сложение CSR матриц."""
        res_coo = self._to_coo() + other._to_coo()
        return res_coo._to_csr()

    def _mul_impl(self, scalar: float) -> 'CSRMatrix':
        """Умножение CSR на скаляр."""
        new_data = [val * scalar for val in self.data]
        return CSRMatrix(new_data, self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Hint:
        Результат - в CSC формате (с теми же данными, но с интерпретацией столбцов как строк).
        """
        from CSC import CSCMatrix
        return CSCMatrix(
            data=list(self.data),
            indices=list(self.indices),
            indptr=list(self.indptr),
            shape=(self.shape[1], self.shape[0])
        )

    def _matmul_impl(self, other: 'Matrix') -> 'CSRMatrix':
        """Умножение CSR матриц."""
        if not isinstance(other, CSRMatrix):
            other = other._to_csr()
        n_rows_a = self.shape[0]
        n_cols_b = other.shape[1]
        res_dense = [[0.0 for _ in range(n_cols_b)] for _ in range(n_rows_a)]
        for i in range(n_rows_a):
            for a_idx in range(self.indptr[i], self.indptr[i + 1]):
                k = self.indices[a_idx]
                val_a = self.data[a_idx]
                for b_idx in range(other.indptr[k], other.indptr[k + 1]):
                    j = other.indices[b_idx]
                    res_dense[i][j] += val_a * other.data[b_idx]
        return CSRMatrix.from_dense(res_dense)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        n_rows = len(dense_matrix)
        n_cols = len(dense_matrix[0]) if n_rows > 0 else 0
        data, indices, indptr = [], [], [0]
        for i in range(n_rows):
            count = 0
            for j in range(n_cols):
                val = dense_matrix[i][j]
                if abs(val) > TOLERANCE:
                    data.append(float(val))
                    indices.append(j)
                    count += 1
            indptr.append(indptr[-1] + count)
        return cls(data, indices, indptr, (n_rows, n_cols))

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
        row_indices, col_indices = [], []
        for i in range(self.shape[0]):
            for idx in range(self.indptr[i], self.indptr[i + 1]):
                row_indices.append(i)
                col_indices.append(self.indices[idx])
        return COOMatrix(list(self.data), row_indices, col_indices, self.shape)