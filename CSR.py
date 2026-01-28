from base import Matrix
from type import *


# УБРАЛ ИМПОРТЫ ОТСЮДА

class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = list(data)
        self.indices = list(indices)
        self.indptr = list(indptr)

    def to_dense(self) -> DenseMatrix:
        n, m = self.shape
        dense = [[0.0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for k in range(self.indptr[i], self.indptr[i + 1]):
                dense[i][self.indices[k]] = self.data[k]
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        # ИМПОРТ ВНУТРИ
        from COO import COOMatrix
        if not isinstance(other, CSRMatrix):
            other = other._to_csr() if hasattr(other, "_to_csr") else CSRMatrix.from_dense(other.to_dense())
        res_coo = self._to_coo()._add_impl(other._to_coo())
        return res_coo._to_csr()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        if scalar == 0:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)
        return CSRMatrix([v * scalar for v in self.data], self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        # ИМПОРТ ВНУТРИ
        from CSC import CSCMatrix
        return CSCMatrix(self.data[:], self.indices[:], self.indptr[:], (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        # ИМПОРТ ВНУТРИ
        from COO import COOMatrix
        return self._to_coo()._matmul_impl(other._to_coo() if hasattr(other, "_to_coo") else other)._to_csr()

    @classmethod
    def from_dense(cls, mtx: DenseMatrix) -> 'CSRMatrix':
        n = len(mtx)
        m = len(mtx[0]) if n > 0 else 0
        v, idx, ptr = [], [], [0]
        for i in range(n):
            for j in range(m):
                if mtx[i][j] != 0:
                    v.append(mtx[i][j]);
                    idx.append(j)
            ptr.append(len(v))
        return cls(v, idx, ptr, (n, m))

    def _to_csc(self) -> 'CSCMatrix':
        # ИМПОРТ ВНУТРИ
        from CSC import CSCMatrix
        return self._to_coo()._to_csc()

    def _to_coo(self) -> 'COOMatrix':
        # ИМПОРТ ВНУТРИ
        from COO import COOMatrix
        n, m = self.shape
        v, r, c = [], [], []
        for i in range(n):
            for k in range(self.indptr[i], self.indptr[i + 1]):
                v.append(self.data[k]);
                r.append(i);
                c.append(self.indices[k])
        return COOMatrix(v, r, c, (n, m))