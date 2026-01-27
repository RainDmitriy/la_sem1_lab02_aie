from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = list(data)
        self.indices = list(indices)
        self.indptr = list(indptr)

    def to_dense(self) -> DenseMatrix:
        r_count, c_count = self.shape
        res = [[0.0 for _ in range(c_count)] for _ in range(r_count)]
        for i in range(r_count):
            for idx in range(self.indptr[i], self.indptr[i + 1]):
                col = self.indices[idx]
                res[i][col] = self.data[idx]
        return res

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        # Самый надежный способ сложения разреженных матриц — через COO
        from COO import COOMatrix
        self_coo = self._to_coo()
        other_coo = other._to_coo() if hasattr(other, '_to_coo') else COOMatrix.from_dense(other.to_dense())
        return (self_coo + other_coo)._to_csr()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        return CSRMatrix([x * scalar for x in self.data], self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование CSR дает CSC с теми же массивами."""
        from CSC import CSCMatrix
        new_shape = (self.shape[1], self.shape[0])
        return CSCMatrix(self.data, self.indices, self.indptr, new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        # Используем алгоритм CSR * Dense для простоты и универсальности
        b_dense = other.to_dense()
        n_rows = self.shape[0]
        n_cols_b = other.shape[1]

        res = [[0.0 for _ in range(n_cols_b)] for _ in range(n_rows)]
        for i in range(n_rows):
            for k_idx in range(self.indptr[i], self.indptr[i + 1]):
                col_a = self.indices[k_idx]
                val_a = self.data[k_idx]
                for j in range(n_cols_b):
                    res[i][j] += val_a * b_dense[col_a][j]

        return CSRMatrix.from_dense(res)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        from COO import COOMatrix
        return COOMatrix.from_dense(dense_matrix)._to_csr()

    def _to_csc(self) -> 'CSCMatrix':
        return self._to_coo()._to_csc()

    def _to_coo(self):
        from COO import COOMatrix
        rows = []
        for i in range(self.shape[0]):
            for _ in range(self.indptr[i], self.indptr[i + 1]):
                rows.append(i)
        return COOMatrix(self.data, rows, self.indices, self.shape)