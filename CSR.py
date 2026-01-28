from base import Matrix
from .types import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix
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
        res = [[0.0 for _ in range(m)] for _ in range(n)]
        for i, (start, end) in enumerate(zip(self.indptr, self.indptr[1:])):
            for k in range(start, end):
                res[i][self.indices[k]] = self.data[k]
        return res

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        if not isinstance(other, CSRMatrix):
            other = other._to_csr() if hasattr(other, "_to_csr") else CSRMatrix.from_dense(other.to_dense())

        # сложение через посредника
        c1 = self._to_coo()
        c2 = other._to_coo()
        return c1._add_impl(c2)._to_csr()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        if scalar == 0:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)
        return CSRMatrix([v * scalar for v in self.data], self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        n, m = self.shape
        # смена формата на CSC
        return CSCMatrix(self.data[:], self.indices[:], self.indptr[:], (m, n))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        from COO import COOMatrix
        a_coo = self._to_coo()
        if isinstance(other, COOMatrix):
            b_coo = other
        else:
            b_coo = other._to_coo() if hasattr(other, "_to_coo") else COOMatrix.from_dense(other.to_dense())
        return a_coo._matmul_impl(b_coo)._to_csr()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        v, idx, ptr = [], [], [0]

        for r in range(rows):
            for c in range(cols):
                val = dense_matrix[r][c]
                if val != 0:
                    v.append(val)
                    idx.append(c)
            ptr.append(len(v))
        return cls(v, idx, ptr, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        return self._to_coo()._to_csc()

    def _to_coo(self) -> 'COOMatrix':
        n, m = self.shape
        v_out, r_out, c_out = [], [], []
        for i in range(n):
            for k in range(self.indptr[i], self.indptr[i + 1]):
                v_out.append(self.data[k])
                r_out.append(i)
                c_out.append(self.indices[k])
        return COOMatrix(v_out, r_out, c_out, (n, m))