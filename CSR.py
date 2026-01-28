from base import Matrix
from my_types import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix
from CSC import CSCMatrix
from COO import COOMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = list(data)
        self.indices = list(indices)
        self.indptr = list(indptr)

    def to_dense(self) -> DenseMatrix:
        n, m = self.shape
        dense = [[0.0] * m for _ in range(n)]
        for i in range(n):
            for k in range(self.indptr[i], self.indptr[i + 1]):
                j = self.indices[k]
                dense[i][j] += self.data[k]
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        if not isinstance(other, CSRMatrix):
            other = other._to_csr() if hasattr(other, "_to_csr") else CSRMatrix.from_dense(other.to_dense())
        coo = self._to_coo()
        coo2 = other._to_coo()
        return coo._add_impl(coo2)._to_csr()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        if scalar == 0:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)
        return CSRMatrix([v * scalar for v in self.data], list(self.indices), list(self.indptr), self.shape)

    def transpose(self) -> 'Matrix':
        n, m = self.shape
        return CSCMatrix(list(self.data), list(self.indices), list(self.indptr), (m, n))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        a = self._to_coo()
        if isinstance(other, COOMatrix):
            b = other
        elif hasattr(other, "_to_coo"):
            b = other._to_coo()
        else:
            b = COOMatrix.from_dense(other.to_dense())
        return a._matmul_impl(b)._to_csr()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        n = len(dense_matrix)
        m = len(dense_matrix[0]) if n > 0 else 0
        data, indices, indptr, count = [], [], [0], 0
        for i in range(n):
            for j in range(m):
                v = dense_matrix[i][j]
                if v != 0:
                    data.append(v)
                    indices.append(j)
                    count += 1
            indptr.append(count)
        return cls(data, indices, indptr, (n, m))

    def _to_csc(self) -> 'CSCMatrix':
        return self._to_coo()._to_csc()

    def _to_coo(self) -> 'COOMatrix':
        n, m = self.shape
        row, col, data = [], [], []
        for i in range(n):
            for k in range(self.indptr[i], self.indptr[i + 1]):
                row.append(i)
                col.append(self.indices[k])
                data.append(self.data[k])
        return COOMatrix(data, row, col, (n, m))