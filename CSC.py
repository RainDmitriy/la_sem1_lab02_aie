from base import Matrix
from type import *


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = list(data)
        self.indices = list(indices)
        self.indptr = list(indptr)

    def to_dense(self) -> DenseMatrix:
        r_cnt, c_cnt = self.shape
        grid = [[0.0 for _ in range(c_cnt)] for _ in range(r_cnt)]
        col_idx = 0
        while col_idx < c_cnt:
            for k in range(self.indptr[col_idx], self.indptr[col_idx + 1]):
                row_idx = self.indices[k]
                grid[row_idx][col_idx] = self.data[k]
            col_idx += 1
        return grid

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        # Импорт внутри метода
        from COO import COOMatrix
        if not isinstance(other, CSCMatrix):
            other = other._to_csc() if hasattr(other, "_to_csc") else CSCMatrix.from_dense(other.to_dense())

        m1_coo = self._to_coo()
        m2_coo = other._to_coo()
        return m1_coo._add_impl(m2_coo)._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        if abs(scalar) < 1e-15:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)
        return CSCMatrix([v * scalar for v in self.data], self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        # Импорт внутри метода
        from CSR import CSRMatrix
        r, c = self.shape
        return CSRMatrix(self.data[:], self.indices[:], self.indptr[:], (c, r))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        # Импорт внутри метода
        from COO import COOMatrix
        left = self._to_coo()
        if isinstance(other, COOMatrix):
            right = other
        else:
            right = other._to_coo() if hasattr(other, "_to_coo") else COOMatrix.from_dense(other.to_dense())
        return left._matmul_impl(right)._to_csc()

    @classmethod
    def from_dense(cls, mtx: DenseMatrix) -> 'CSCMatrix':
        r_num = len(mtx)
        c_num = len(mtx[0]) if r_num > 0 else 0
        v, idx, ptr = [], [], [0]
        for j in range(c_num):
            for i in range(r_num):
                if mtx[i][j] != 0:
                    v.append(mtx[i][j])
                    idx.append(i)
            ptr.append(len(v))
        return cls(v, idx, ptr, (r_num, c_num))

    def _to_csr(self) -> 'CSRMatrix':
        return self._to_coo()._to_csr()

    def _to_coo(self) -> 'COOMatrix':
        from COO import COOMatrix
        r_tot, c_tot = self.shape
        v_out, r_out, c_out = [], [], []
        for j in range(c_tot):
            for k in range(self.indptr[j], self.indptr[j + 1]):
                v_out.append(self.data[k])
                r_out.append(self.indices[k])
                c_out.append(j)
        return COOMatrix(v_out, r_out, c_out, (r_tot, c_tot))