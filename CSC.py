from base import Matrix
from .types import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix
from CSR import CSRMatrix
from COO import COOMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = list(data)
        self.indices = list(indices)
        self.indptr = list(indptr)

    def to_dense(self) -> DenseMatrix:
        rows, cols = self.shape
        res = [[0.0 for _ in range(cols)] for _ in range(rows)]
        c = 0
        while c < cols:
            for idx in range(self.indptr[c], self.indptr[c + 1]):
                r = self.indices[idx]
                res[r][c] = self.data[idx]
            c += 1
        return res

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        if not isinstance(other, CSCMatrix):
            other = other._to_csc() if hasattr(other, "_to_csc") else CSCMatrix.from_dense(other.to_dense())

        # через coo
        m1 = self._to_coo()
        m2 = other._to_coo()
        return m1._add_impl(m2)._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        if abs(scalar) < 1e-18:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)
        new_v = [val * scalar for val in self.data]
        return CSCMatrix(new_v, self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        r, c = self.shape
        # инверсия формата
        return CSRMatrix(self.data[:], self.indices[:], self.indptr[:], (c, r))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
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
        vals, idxs, ptr = [], [], [0]

        for j in range(c_num):
            for i in range(r_num):
                if mtx[i][j] != 0:
                    vals.append(mtx[i][j])
                    idxs.append(i)
            ptr.append(len(vals))
        return cls(vals, idxs, ptr, (r_num, c_num))

    def _to_csr(self) -> 'CSRMatrix':
        return self._to_coo()._to_csr()

    def _to_coo(self) -> 'COOMatrix':
        r_total, c_total = self.shape
        v_list, r_list, c_list = [], [], []
        for j in range(c_total):
            for k in range(self.indptr[j], self.indptr[j + 1]):
                v_list.append(self.data[k])
                r_list.append(self.indices[k])
                c_list.append(j)
        return COOMatrix(v_list, r_list, c_list, (r_total, c_total))