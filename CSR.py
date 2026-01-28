from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from CSC import CSCMatrix
    from COO import COOMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)

        if len(indptr) != shape[0] + 1:
            raise ValueError(f"Неверная длина indptr")
        if indptr[0] != 0:
            raise ValueError("indptr[0] должен быть 0")
        if indptr[-1] != len(data):
            raise ValueError(f"Несоответствие indptr[-1]")
        if len(data) != len(indices):
            raise ValueError("Разные длины массивов")

        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        r_cnt, c_cnt = self.shape

        res_mat = [[0] * c_cnt for _ in range(r_cnt)]

        for i in range(r_cnt):
            st = self.indptr[i]
            en = self.indptr[i + 1]
            for pos in range(st, en):
                j = self.indices[pos]
                v = self.data[pos]
                res_mat[i][j] = v

        return res_mat

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        coo_a = self._to_coo()
        coo_b = other._to_coo()

        coo_res = coo_a._add_impl(coo_b)

        return coo_res._to_csr()

    def _mul_impl(self, k: float) -> 'Matrix':
        if k == 0:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)

        new_vals = [x * k for x in self.data]

        return CSRMatrix(new_vals, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        from CSC import CSCMatrix

        r_cnt, c_cnt = self.shape
        nr_cnt, nc_cnt = c_cnt, r_cnt

        col_cnts = [0] * nc_cnt

        for i in range(r_cnt):
            st = self.indptr[i]
            en = self.indptr[i + 1]
            col_cnts[i] = en - st

        new_ptr = [0] * (nc_cnt + 1)
        for j in range(nc_cnt):
            new_ptr[j + 1] = new_ptr[j] + col_cnts[j]

        new_vals = [0] * len(self.data)
        new_idxs = [0] * len(self.indices)

        col_pos = new_ptr.copy()

        for i in range(r_cnt):
            st = self.indptr[i]
            en = self.indptr[i + 1]

            for pos in range(st, en):
                j = self.indices[pos]
                v = self.data[pos]

                p = col_pos[i]
                new_vals[p] = v
                new_idxs[p] = j
                col_pos[i] += 1

        return CSCMatrix(new_vals, new_idxs, new_ptr, (nr_cnt, nc_cnt))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        ra, ca = self.shape
        rb, cb = other.shape

        res_vals = []
        res_idxs = []
        res_ptr = [0] * (ra + 1)

        for i_idx in range(ra):
            row_sum = {}

            a_st = self.indptr[i_idx]
            a_en = self.indptr[i_idx + 1]

            for a_pos in range(a_st, a_en):
                k_idx = self.indices[a_pos]
                a_v = self.data[a_pos]

                b_st = other.indptr[k_idx]
                b_en = other.indptr[k_idx + 1]

                for b_pos in range(b_st, b_en):
                    j_idx = other.indices[b_pos]
                    b_v = other.data[b_pos]

                    if j_idx not in row_sum:
                        row_sum[j_idx] = 0.0
                    row_sum[j_idx] += a_v * b_v

            sorted_cols = sorted(row_sum.keys())
            for j_idx in sorted_cols:
                v_val = row_sum[j_idx]
                if abs(v_val) > 1e-14:
                    res_vals.append(v_val)
                    res_idxs.append(j_idx)

            res_ptr[i_idx + 1] = len(res_vals)

        return CSRMatrix(res_vals, res_idxs, res_ptr, (ra, cb))

    @classmethod
    def from_dense(cls, dense_mat: DenseMatrix) -> 'CSRMatrix':
        r_cnt = len(dense_mat)
        c_cnt = len(dense_mat[0])

        vals = []
        idxs = []

        row_cnts = [0] * r_cnt

        for i in range(r_cnt):
            for j in range(c_cnt):
                v = dense_mat[i][j]
                if v != 0:
                    vals.append(v)
                    idxs.append(j)
                    row_cnts[i] += 1

        ptrs = [0] * (r_cnt + 1)
        for i in range(r_cnt):
            ptrs[i + 1] = ptrs[i] + row_cnts[i]

        return cls(vals, idxs, ptrs, (r_cnt, c_cnt))

    def _to_csc(self) -> 'CSCMatrix':
        from CSC import CSCMatrix

        m, n = self.shape

        col_cnts = [0] * n

        for col_idx in self.indices:
            col_cnts[col_idx] += 1

        new_ptr = [0] * (n + 1)

        for j in range(n):
            new_ptr[j + 1] = new_ptr[j] + col_cnts[j]

        new_vals = [0] * len(self.data)
        new_idxs = [0] * len(self.indices)

        cur_pos = new_ptr.copy()

        for i in range(m):
            row_st = self.indptr[i]
            row_en = self.indptr[i + 1]

            for k in range(row_st, row_en):
                j_val = self.indices[k]
                v_val = self.data[k]

                p_val = cur_pos[j_val]

                new_vals[p_val] = v_val
                new_idxs[p_val] = i

                cur_pos[j_val] += 1

        return CSCMatrix(new_vals, new_idxs, new_ptr, (m, n))

    def _to_coo(self) -> 'COOMatrix':
        from COO import COOMatrix

        r_cnt, c_cnt = self.shape

        vals = []
        rows = []
        cols = []

        for i in range(r_cnt):
            st = self.indptr[i]
            en = self.indptr[i + 1]

            for pos in range(st, en):
                j = self.indices[pos]
                v = self.data[pos]

                vals.append(v)
                rows.append(i)
                cols.append(j)

        return COOMatrix(vals, rows, cols, self.shape)