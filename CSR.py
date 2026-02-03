from base import Matrix
from types import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        dense = [[0.0 for _ in range(self.shape[1])] for _ in range(self.shape[0])]
        for i in range(self.shape[0]):
            for j in range(self.indptr[i], self.indptr[i + 1]):
                dense[i][self.indices[j]] = self.data[j]
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        return self._to_coo()._add_impl(other._to_coo())._to_csr()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        return CSRMatrix([v * scalar for v in self.data], self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Hint:
        Результат - в CSC формате (с теми же данными, но с интерпретацией столбцов как строк).
        """
        from CSC import CSCMatrix
        return CSCMatrix(self.data[:], self.indices[:], self.indptr[:], (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        other_csr = other._to_csr()
        res_rows, res_cols, res_data = [], [], []
        for i in range(self.shape[0]):
            row_map = {}
            for j in range(self.indptr[i], self.indptr[i + 1]):
                a_val = self.data[j]
                a_col = self.indices[j]
                for k in range(other_csr.indptr[a_col], other_csr.indptr[a_col + 1]):
                    b_col = other_csr.indices[k]
                    b_val = other_csr.data[k]
                    row_map[b_col] = row_map.get(b_col, 0.0) + a_val * b_val
            for c in sorted(row_map.keys()):
                if row_map[c] != 0:
                    res_rows.append(i)
                    res_cols.append(c)
                    res_data.append(row_map[c])
        from COO import COOMatrix
        return COOMatrix(res_data, res_rows, res_cols, (self.shape[0], other.shape[1]))._to_csr()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        from COO import COOMatrix
        return COOMatrix.from_dense(dense_matrix)._to_csr()

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSRMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix
        return self._to_coo()._to_csc()
    
    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSRMatrix в COOMatrix.
        """
        from COO import COOMatrix
        rows = []
        for i in range(self.shape[0]):
            for j in range(self.indptr[i], self.indptr[i + 1]):
                rows.append(i)
        return COOMatrix(self.data[:], rows, self.indices[:], self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        return self
