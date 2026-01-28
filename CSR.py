from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from CSC import CSCMatrix
    from COO import COOMatrix


class CSRMatrix(Matrix):
    def __init__(self, d: CSRData, idxs: CSRIndices, ptrs: CSRIndptr, s: Shape):
        super().__init__(s)

        if len(ptrs) != s[0] + 1:
            raise ValueError(f"Неверная длина ptrs")
        if ptrs[0] != 0:
            raise ValueError("ptrs[0] должен быть 0")
        if ptrs[-1] != len(d):
            raise ValueError(f"Несоответствие ptrs[-1]")
        if len(d) != len(idxs):
            raise ValueError("Разные длины массивов")
        
        self.d = d
        self.idxs = idxs
        self.ptrs = ptrs

    def to_dense(self) -> DenseMatrix:
        rows_cnt, cols_cnt = self.s

        result_mat = [[0] * cols_cnt for _ in range(rows_cnt)]
        
        for i in range(rows_cnt):
            st = self.ptrs[i]
            en = self.ptrs[i + 1]
            for pos in range(st, en):

                j = self.idxs[pos]
                v = self.d[pos]
                result_mat[i][j] = v
        
        return result_mat

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        coo_a = self._to_coo()
        coo_b = other._to_coo()
        
        coo_res = coo_a._add_impl(coo_b)
        
        return coo_res._to_csr()

    def _mul_impl(self, k: float) -> 'Matrix':
        if k == 0:
            return CSRMatrix([], [], [0] * (self.s[0] + 1), self.s)
        
        new_vals = [x * k for x in self.d]

        return CSRMatrix(new_vals, self.idxs.copy(), self.ptrs.copy(), self.s)

    def transpose(self) -> 'Matrix':
        from CSC import CSCMatrix
        
        rows_cnt, cols_cnt = self.s
        new_r_cnt, new_c_cnt = cols_cnt, rows_cnt
        
        cnts_per_col = [0] * new_c_cnt
        
        for i in range(rows_cnt):
            st = self.ptrs[i]
            en = self.ptrs[i + 1]
            cnts_per_col[i] = en - st
        
        new_ptrs = [0] * (new_c_cnt + 1)
        for j in range(new_c_cnt):
            new_ptrs[j + 1] = new_ptrs[j] + cnts_per_col[j]
        
        new_vals = [0] * len(self.d)
        new_idxs = [0] * len(self.idxs)
        
        pos_per_col = new_ptrs.copy()
        
        for i in range(rows_cnt):
            st = self.ptrs[i]
            en = self.ptrs[i + 1]
            
            for pos in range(st, en):
                j = self.idxs[pos]
                v = self.d[pos]
                
                p = pos_per_col[i]
                new_vals[p] = v
                new_idxs[p] = j
                pos_per_col[i] += 1
        
        return CSCMatrix(new_vals, new_idxs, new_ptrs, (new_r_cnt, new_c_cnt))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        rows_a, cols_a = self.s
        rows_b, cols_b = other.s

        res_vals = []
        res_idxs = []
        res_ptrs = [0] * (rows_a + 1)

        for i_idx in range(rows_a):
            row_sum = {}

            a_st = self.ptrs[i_idx]
            a_en = self.ptrs[i_idx + 1]

            for a_pos in range(a_st, a_en):
                k_idx = self.idxs[a_pos]
                a_v = self.d[a_pos]

                b_st = other.ptrs[k_idx]
                b_en = other.ptrs[k_idx + 1]

                for b_pos in range(b_st, b_en):
                    j_idx = other.idxs[b_pos]
                    b_v = other.d[b_pos]

                    if j_idx not in row_sum:
                        row_sum[j_idx] = 0.0
                    row_sum[j_idx] += a_v * b_v

            sorted_cols = sorted(row_sum.keys())
            for j_idx in sorted_cols:
                v_val = row_sum[j_idx]
                if abs(v_val) > 1e-14:
                    res_vals.append(v_val)
                    res_idxs.append(j_idx)

            res_ptrs[i_idx + 1] = len(res_vals)

        return CSRMatrix(res_vals, res_idxs, res_ptrs, (rows_a, cols_b))

    @classmethod
    def from_dense(cls, dense_mat: DenseMatrix) -> 'CSRMatrix':
        rows_cnt = len(dense_mat)
        cols_cnt = len(dense_mat[0])
        
        vals = []
        idxs = []
        
        cnts_per_row = [0] * rows_cnt
        
        for i in range(rows_cnt):
            for j in range(cols_cnt):
                v = dense_mat[i][j]
                if v != 0:
                    vals.append(v)
                    idxs.append(j)
                    cnts_per_row[i] += 1
        
        ptrs = [0] * (rows_cnt + 1)
        for i in range(rows_cnt):
            ptrs[i + 1] = ptrs[i] + cnts_per_row[i]
        
        return cls(vals, idxs, ptrs, (rows_cnt, cols_cnt))

    def _to_csc(self) -> 'CSCMatrix':
        from CSC import CSCMatrix

        m_val, n_val = self.s

        cnts_per_col = [0] * n_val

        for col_idx in self.idxs:
            cnts_per_col[col_idx] += 1

        new_ptrs = [0] * (n_val + 1)

        for j in range(n_val):
            new_ptrs[j + 1] = new_ptrs[j] + cnts_per_col[j]

        new_vals = [0] * len(self.d)
        new_idxs = [0] * len(self.idxs)

        cur_pos = new_ptrs.copy()

        for i in range(m_val):
            row_st = self.ptrs[i]
            row_en = self.ptrs[i + 1]

            for k in range(row_st, row_en):
                j_val = self.idxs[k]
                v_val = self.d[k]

                p_val = cur_pos[j_val]

                new_vals[p_val] = v_val
                new_idxs[p_val] = i

                cur_pos[j_val] += 1
    
        return CSCMatrix(new_vals, new_idxs, new_ptrs, (m_val, n_val))
    
    def _to_coo(self) -> 'COOMatrix':
        from COO import COOMatrix

        rows_cnt, cols_cnt = self.s

        vals = []
        row_idxs = []
        col_idxs = []
        
        for i in range(rows_cnt):
            st = self.ptrs[i]
            en = self.ptrs[i + 1]
            
            for pos in range(st, en):
                j = self.idxs[pos]
                v = self.d[pos]
                
                vals.append(v)
                row_idxs.append(i)
                col_idxs.append(j)
        
        return COOMatrix(vals, row_idxs, col_idxs, self.s)