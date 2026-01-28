from base import Matrix
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix, TOLERANCE

class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу."""
        n_rows, n_cols = self.shape
        dense = [[0.0 for _ in range(n_cols)] for _ in range(n_rows)]
        for j in range(n_cols):
            for idx in range(self.indptr[j], self.indptr[j + 1]):
                row_idx = self.indices[idx]
                dense[row_idx][j] = self.data[idx]
        return dense

    def _add_impl(self, other: 'CSCMatrix') -> 'CSCMatrix':
        """Сложение CSC матриц."""
        res_coo = self._to_coo() + other._to_coo()
        return res_coo._to_csc()

    def _mul_impl(self, scalar: float) -> 'CSCMatrix':
        """Умножение CSC на скаляр."""
        new_data = [val * scalar for val in self.data]
        return CSCMatrix(new_data, self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Hint:
        Результат - в CSR формате (с теми же данными, но с интерпретацией строк как столбцов).
        """
        from CSR import CSRMatrix
        return CSRMatrix(
            data=list(self.data),
            indices=list(self.indices),
            indptr=list(self.indptr),
            shape=(self.shape[1], self.shape[0])
        )

    def _matmul_impl(self, other: 'Matrix') -> 'CSCMatrix':
        """Умножение CSC матриц."""
        if not isinstance(other, CSCMatrix):
            other = other._to_csc()
        n_rows_a = self.shape[0]
        n_cols_b = other.shape[1]
        res_dense = [[0.0 for _ in range(n_cols_b)] for _ in range(n_rows_a)]
        for j in range(n_cols_b):
            for k_idx in range(other.indptr[j], other.indptr[j+1]):
                k = other.indices[k_idx]
                val_b = other.data[k_idx]
                for a_idx in range(self.indptr[k], self.indptr[k+1]):
                    i = self.indices[a_idx]
                    res_dense[i][j] += self.data[a_idx] * val_b
        return CSCMatrix.from_dense(res_dense)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        n_rows = len(dense_matrix)
        n_cols = len(dense_matrix[0]) if n_rows > 0 else 0
        data, indices, indptr = [], [], [0]
        for j in range(n_cols):
            count = 0
            for i in range(n_rows):
                val = dense_matrix[i][j]
                if abs(val) > TOLERANCE:
                    data.append(float(val))
                    indices.append(i)
                    count += 1
            indptr.append(indptr[-1] + count)
        return cls(data, indices, indptr, (n_rows, n_cols))

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
        row_indices, col_indices = [], []
        for j in range(self.shape[1]):
            for idx in range(self.indptr[j], self.indptr[j+1]):
                row_indices.append(self.indices[idx])
                col_indices.append(j)
        return COOMatrix(list(self.data), row_indices, col_indices, self.shape)