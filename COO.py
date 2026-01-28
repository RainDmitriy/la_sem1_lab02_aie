from base import Matrix
from type import *

class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.data, self.row, self.col = list(data), list(row), list(col)

    def to_dense(self) -> DenseMatrix:
        r, c = self.shape
        res = [[0.0 for _ in range(c)] for _ in range(r)]
        for v, i, j in zip(self.data, self.row, self.col):
            res[i][j] += v
        return res

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        if not isinstance(other, COOMatrix):
            other = other._to_coo() if hasattr(other, "_to_coo") else COOMatrix.from_dense(other.to_dense())
        d = {}
        for v, r, c in zip(self.data, self.row, self.col): d[(r,c)] = d.get((r,c), 0)+v
        for v, r, c in zip(other.data, other.row, other.col): d[(r,c)] = d.get((r,c), 0)+v
        vals, rs, cs = [], [], []
        for (r,c), v in d.items():
            if v != 0: vals.append(v); rs.append(r); cs.append(c)
        return COOMatrix(vals, rs, cs, self.shape)

    def _mul_impl(self, val: float) -> 'Matrix':
        if val == 0: return COOMatrix([], [], [], self.shape)
        return COOMatrix([v * val for v in self.data], self.row[:], self.col[:], self.shape)

    def transpose(self) -> 'Matrix':
        r, c = self.shape
        return COOMatrix(self.data[:], self.col[:], self.row[:], (c, r))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        if not isinstance(other, COOMatrix):
            other = other._to_coo() if hasattr(other, "_to_coo") else COOMatrix.from_dense(other.to_dense())
        b_map = {}
        for v, r, c in zip(other.data, other.row, other.col):
            b_map.setdefault(r, []).append((c, v))
        res = {}
        for v, r, c in zip(self.data, self.row, self.col):
            if c in b_map:
                for oc, ov in b_map[c]:
                    res[(r, oc)] = res.get((r, oc), 0) + v * ov
        nv, nr, nc = [], [], []
        for (r, c), v in res.items():
            if v != 0: nv.append(v); nr.append(r); nc.append(c)
        return COOMatrix(nv, nr, nc, (self.shape[0], other.shape[1]))

    @classmethod
    def from_dense(cls, mtx: DenseMatrix) -> 'COOMatrix':
        r_c = len(mtx)
        c_c = len(mtx[0]) if r_c > 0 else 0
        v, r, c = [], [], []
        for i in range(r_c):
            for j in range(c_c):
                if mtx[i][j] != 0: v.append(mtx[i][j]); r.append(i); c.append(j)
        return cls(v, r, c, (r_c, c_c))

    def _to_csc(self) -> 'CSCMatrix':
        from CSC import CSCMatrix  # Внутренний импорт
        n, m = self.shape
        counts = [0] * m
        for c in self.col: counts[c] += 1
        ptr = [0] * (m + 1)
        for j in range(m): ptr[j+1] = ptr[j] + counts[j]
        cur, d, idx = ptr[:-1].copy(), [0.0]*len(self.data), [0]*len(self.data)
        for v, r, c in zip(self.data, self.row, self.col):
            p = cur[c]; d[p], idx[p] = v, r; cur[c] += 1
        return CSCMatrix(d, idx, ptr, (n, m))

    def _to_csr(self) -> 'CSRMatrix':
        from CSR import CSRMatrix  # Внутренний импорт
        n, m = self.shape
        counts = [0] * n
        for r in self.row: counts[r] += 1
        ptr = [0] * (n + 1)
        for i in range(n): ptr[i+1] = ptr[i] + counts[i]
        cur, d, idx = ptr[:-1].copy(), [0.0]*len(self.data), [0]*len(self.data)
        for v, r, c in zip(self.data, self.row, self.col):
            p = cur[r]; d[p], idx[p] = v, c; cur[r] += 1
        return CSRMatrix(d, idx, ptr, (n, m))