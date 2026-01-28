from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix
from typing import Dict, Tuple, List

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from CSC import CSCMatrix
    from CSR import CSRMatrix


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)

        if not (len(data) == len(row) == len(col)):
            raise ValueError("Разные длины массивов")

        self.data = data
        self.row = row
        self.col = col
        self.shape = shape

    def to_dense(self) -> DenseMatrix:
        r_cnt, c_cnt = self.shape
        res = []

        for i in range(r_cnt):
            r_list = [0] * c_cnt
            res.append(r_list)

        for idx in range(len(self.data)):
            res[self.row[idx]][self.col[idx]] = self.data[idx]

        return res

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        values_map: Dict[Tuple[int, int], float] = {}

        for v, r_idx, c_idx in zip(self.data, self.row, self.col):
            values_map[(r_idx, c_idx)] = values_map.get((r_idx, c_idx), 0.0) + v

        for v, r_idx, c_idx in zip(other.data, other.row, other.col):
            values_map[(r_idx, c_idx)] = values_map.get((r_idx, c_idx), 0.0) + v

        nd, nr, nc = [], [], []
        for (r_idx, c_idx), val in values_map.items():
            if val != 0:
                nd.append(val)
                nr.append(r_idx)
                nc.append(c_idx)

        return COOMatrix(nd, nr, nc, self.shape)

    def _mul_impl(self, k: float) -> 'Matrix':
        new_vals = [x * k for x in self.data]
        return COOMatrix(new_vals, self.row[:], self.col[:], self.shape)

    def transpose(self) -> 'Matrix':
        ns = (self.shape[1], self.shape[0])
        return COOMatrix(self.data.copy(), self.col.copy(), self.row.copy(), ns)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        if self.shape[1] != other.shape[0]:
            raise ValueError("Некорректные размеры")

        m, n = self.shape[0], other.shape[1]
        res_map = {}

        for idx in range(len(self.data)):
            ra = self.row[idx]
            ca = self.col[idx]
            va = self.data[idx]

            csr_other = other._to_csr()
            st = csr_other.indptr[ca]
            en = csr_other.indptr[ca + 1]
            for k_idx in range(st, en):
                cb = csr_other.indices[k_idx]
                vb = csr_other.data[k_idx]
                key = (ra, cb)
                res_map[key] = res_map.get(key, 0.0) + va * vb

        fd, fr, fc = [], [], []
        for (i_idx, j_idx), v in res_map.items():
            if abs(v) > 1e-14:
                fd.append(v)
                fr.append(i_idx)
                fc.append(j_idx)

        return COOMatrix(fd, fr, fc, (m, n))

    @classmethod
    def from_dense(cls, dense_mat: DenseMatrix) -> 'COOMatrix':
        vals, rows, cols = [], [], []

        for i_idx in range(len(dense_mat)):
            for j_idx in range(len(dense_mat[0])):
                v = dense_mat[i_idx][j_idx]
                if v != 0:
                    vals.append(v)
                    rows.append(i_idx)
                    cols.append(j_idx)

        return COOMatrix(vals, rows, cols, (len(dense_mat), len(dense_mat[0])))

    def _to_csc(self) -> 'CSCMatrix':
        from CSC import CSCMatrix
        r_cnt, c_cnt = self.shape

        items = list(zip(self.col, self.row, self.data))
        items.sort()

        d_vals: List[float] = []
        idxs: List[int] = []
        ptrs: List[int] = [0] * (c_cnt + 1)

        for c_idx, r_idx, v in items:
            d_vals.append(v)
            idxs.append(r_idx)
            ptrs[c_idx + 1] += 1

        for j in range(c_cnt):
            ptrs[j + 1] += ptrs[j]

        return CSCMatrix(d_vals, idxs, ptrs, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        from CSR import CSRMatrix

        r_cnt, c_cnt = self.shape

        items = list(zip(self.row, self.col, self.data))
        items.sort()

        d_vals: List[float] = []
        idxs: List[int] = []
        ptrs: List[int] = [0] * (r_cnt + 1)

        for r_idx, c_idx, v in items:
            d_vals.append(v)
            idxs.append(c_idx)
            ptrs[r_idx + 1] += 1

        for i in range(r_cnt):
            ptrs[i + 1] += ptrs[i]

        return CSRMatrix(d_vals, idxs, ptrs, self.shape)