from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix
from typing import Dict, Tuple, List

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from CSC import CSCMatrix
    from CSR import CSRMatrix

class COOMatrix(Matrix):
    def __init__(self, d: COOData, r: COORows, c: COOCols, s: Shape):
        super().__init__(s)

        if not (len(d) == len(r) == len(c)):
            raise ValueError("Размеры массивов не совпадают")
        
        self.d = d
        self.r = r
        self.c = c
        self.s = s

    def to_dense(self) -> DenseMatrix:
        rows_cnt, cols_cnt = self.s
        result = []

        for i in range(rows_cnt):
            row_list = [0] * cols_cnt
            result.append(row_list)

        for idx in range(len(self.d)):
            result[ self.r[idx] ][ self.c[idx] ] = self.d[idx]
    
        return result

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        values_map: Dict[Tuple[int, int], float] = {}
        
        for v, row_idx, col_idx in zip(self.d, self.r, self.c):
            values_map[(row_idx, col_idx)] = values_map.get((row_idx, col_idx), 0.0) + v
        
        for v, row_idx, col_idx in zip(other.d, other.r, other.c):
            values_map[(row_idx, col_idx)] = values_map.get((row_idx, col_idx), 0.0) + v
        
        new_d, new_r, new_c = [], [], []
        for (row_idx, col_idx), val in values_map.items():
            if val != 0:
                new_d.append(val)
                new_r.append(row_idx)
                new_c.append(col_idx)
        
        return COOMatrix(new_d, new_r, new_c, self.s)

    def _mul_impl(self, k: float) -> 'Matrix':
        new_values = [x * k for x in self.d]
        return COOMatrix(new_values, self.r[:], self.c[:], self.s)

    def transpose(self) -> 'Matrix':
        new_s = (self.s[1], self.s[0])
        return COOMatrix(self.d.copy(), self.c.copy(), self.r.copy(), new_s)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        if self.s[1] != other.s[0]:
            raise ValueError("Недопустимые размеры матриц")

        m, n = self.s[0], other.s[1]
        res_map = {}

        for idx in range(len(self.d)):
            ra = self.r[idx]
            ca = self.c[idx]
            va = self.d[idx]

            other_csr = other._to_csr()
            start_idx = other_csr.indptr[ca]
            end_idx = other_csr.indptr[ca + 1]
            for k_idx in range(start_idx, end_idx):
                cb = other_csr.indices[k_idx]
                vb = other_csr.data[k_idx]
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
        rows_cnt, cols_cnt = self.s
        
        items = list(zip(self.c, self.r, self.d))
        items.sort()
        
        d_vals: List[float] = []
        idxs: List[int] = []
        ptrs: List[int] = [0] * (cols_cnt + 1)
        
        for col_idx, row_idx, v in items:
            d_vals.append(v)
            idxs.append(row_idx)
            ptrs[col_idx + 1] += 1
        
        for j in range(cols_cnt):
            ptrs[j + 1] += ptrs[j]
        
        return CSCMatrix(d_vals, idxs, ptrs, self.s)

    def _to_csr(self) -> 'CSRMatrix':
        from CSR import CSRMatrix

        rows_cnt, cols_cnt = self.s
        
        items = list(zip(self.r, self.c, self.d))
        items.sort()
        
        d_vals: List[float] = []
        idxs: List[int] = []
        ptrs: List[int] = [0] * (rows_cnt + 1)
        
        for row_idx, col_idx, v in items:
            d_vals.append(v)
            idxs.append(col_idx)
            ptrs[row_idx + 1] += 1
        
        for i in range(rows_cnt):
            ptrs[i + 1] += ptrs[i]
        
        return CSRMatrix(d_vals, idxs, ptrs, self.s)