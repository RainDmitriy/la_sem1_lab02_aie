# CSC.py
from base import Matrix
from my_types import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix

class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data, self.indices, self.indptr = list(data), list(indices), list(indptr)

    def to_dense(self) -> DenseMatrix:
        n, m = self.shape
        dense = [[0.0] * m for _ in range(n)]
        for j in range(m):
            for k in range(self.indptr[j], self.indptr[j + 1]):
                dense[self.indices[k]][j] += self.data[k]
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        if not hasattr(other, "_to_csc"):
            other = CSCMatrix.from_dense(other.to_dense())
        else:
            other = other._to_csc()
        return self._to_coo()._add_impl(other._to_coo())._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        if scalar == 0: return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)
        return CSCMatrix([v * scalar for v in self.data], self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        from CSR import CSRMatrix
        return CSRMatrix(self.data[:], self.indices[:], self.indptr[:], (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        from COO import COOMatrix
        a_coo = self._to_coo()
        b_coo = other._to_coo() if hasattr(other, "_to_coo") else COOMatrix.from_dense(other.to_dense())
        return a_coo._matmul_impl(b_coo)._to_csc()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        n, m = len(dense_matrix), len(dense_matrix[0]) if dense_matrix else 0
        data, indices, indptr, cnt = [], [], [0], 0
        for j in range(m):
            for i in range(n):
                if dense_matrix[i][j] != 0:
                    data.append(dense_matrix[i][j])
                    indices.append(i)
                    cnt += 1
            indptr.append(cnt)
        return cls(data, indices, indptr, (n, m))

    def _to_csr(self) -> 'CSRMatrix':
        return self._to_coo()._to_csr()

    def _to_coo(self) -> 'COOMatrix':
        from COO import COOMatrix
        n, m = self.shape
        r, c, d = [], [], []
        for j in range(m):
            for k in range(self.indptr[j], self.indptr[j+1]):
                r.append(self.indices[k])
                c.append(j)
                d.append(self.data[k])
        return COOMatrix(d, r, c, (n, m))