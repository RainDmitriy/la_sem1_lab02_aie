from base import Matrix
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = list(data)
        self.indices = list(indices)
        self.indptr = list(indptr)

    def to_dense(self) -> DenseMatrix:
        r_count, c_count = self.shape
        res = [[0.0 for _ in range(c_count)] for _ in range(r_count)]
        for j in range(c_count):
            for idx in range(self.indptr[j], self.indptr[j + 1]):
                row = self.indices[idx]
                res[row][j] = self.data[idx]
        return res

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        from COO import COOMatrix
        self_coo = self._to_coo()
        other_coo = other._to_coo() if hasattr(other, '_to_coo') else COOMatrix.from_dense(other.to_dense())
        return (self_coo + other_coo)._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        return CSCMatrix([x * scalar for x in self.data], self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        from CSR import CSRMatrix
        new_shape = (self.shape[1], self.shape[0])
        return CSRMatrix(self.data, self.indices, self.indptr, new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        # Перемножаем через плотный формат
        a_dense = self.to_dense()
        b_dense = other.to_dense()
        r_a, c_a = self.shape
        c_b = other.shape[1]

        res = [[0.0 for _ in range(c_b)] for _ in range(r_a)]
        for i in range(r_a):
            for k in range(c_a):
                if a_dense[i][k] != 0:
                    for j in range(c_b):
                        res[i][j] += a_dense[i][k] * b_dense[k][j]
        return CSCMatrix.from_dense(res)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        from COO import COOMatrix
        return COOMatrix.from_dense(dense_matrix)._to_csc()

    def _to_csr(self) -> 'CSRMatrix':
        return self._to_coo()._to_csr()

    def _to_coo(self):
        from COO import COOMatrix
        cols = []
        for j in range(self.shape[1]):
            for _ in range(self.indptr[j], self.indptr[j + 1]):
                cols.append(j)
        return COOMatrix(self.data, self.indices, cols, self.shape)