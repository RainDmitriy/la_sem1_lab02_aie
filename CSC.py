from base import Matrix
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from COO import COOMatrix
    from CSR import CSRMatrix

class CSCMatrix(Matrix):
    def __init__(self, d: CSCData, idxs: CSCIndices, ptrs: CSCIndptr, s: Shape):
        super().__init__(s)

        if len(ptrs) != s[1] + 1:
            raise ValueError(f"Неверная длина ptrs")
        if ptrs[0] != 0:
            raise ValueError("ptrs[0] должен быть нулем")
        if ptrs[-1] != len(d):
            raise ValueError(f"Несоответствие ptrs[-1]")
        if len(d) != len(idxs):
            raise ValueError(f"Разные длины массивов")
        
        self.d = d
        self.idxs = idxs
        self.ptrs = ptrs

    def to_dense(self) -> DenseMatrix:
        rows_cnt, cols_cnt = self.s

        result_mat = [[0] * cols_cnt for _ in range(rows_cnt)]
        
        for j in range(cols_cnt):
            start_p = self.ptrs[j]
            end_p = self.ptrs[j + 1]

            for pos in range(start_p, end_p):
                i = self.idxs[pos]
                v = self.d[pos]
                result_mat[i][j] = v
        
        return result_mat

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        coo_a = self._to_coo()
        coo_b = other._to_coo()
        coo_res = coo_a._add_impl(coo_b)
        
        return coo_res._to_csc()

    def _mul_impl(self, k: float) -> 'Matrix':
        if k == 0:
            return CSCMatrix([], [], [0] * (self.s[1] + 1), self.s)
        
        new_vals = [x * k for x in self.d]
        
        return CSCMatrix(new_vals, self.idxs.copy(), self.ptrs.copy(), self.s)

    def transpose(self) -> 'Matrix':
        from CSR import CSRMatrix

        rows_cnt, cols_cnt = self.s
        new_r_cnt, new_c_cnt = cols_cnt, rows_cnt
        
        cnts_per_row = [0] * new_r_cnt
        
        for j in range(cols_cnt):
            st = self.ptrs[j]
            en = self.ptrs[j + 1]
            cnts_per_row[j] = en - st
        
        new_ptrs = [0] * (new_r_cnt + 1)
        for i in range(new_r_cnt):
            new_ptrs[i + 1] = new_ptrs[i] + cnts_per_row[i]
        
        new_d = [0] * len(self.d)
        new_idxs = [0] * len(self.idxs)
        
        pos_per_row = new_ptrs.copy()
        
        for j in range(cols_cnt):
            st = self.ptrs[j]
            en = self.ptrs[j + 1]
            
            for pos in range(st, en):
                i = self.idxs[pos]
                v = self.d[pos]
                
                p = pos_per_row[j]
                new_d[p] = v
                new_idxs[p] = i
                pos_per_row[j] += 1
        
        return CSRMatrix(new_d, new_idxs, new_ptrs, (new_r_cnt, new_c_cnt))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        rows_a, cols_a = self.s
        rows_b, cols_b = other.s

        res_d = []
        res_idxs = []
        res_ptrs = [0] * (cols_b + 1)

        rows_b = other.s[0]
        cols_b = other.s[1]

        row_entries_b = [[] for _ in range(rows_b)]

        for col_idx in range(cols_b):
            st = other.ptrs[col_idx]
            en = other.ptrs[col_idx + 1]
            for pos in range(st, en):
                row_idx = other.idxs[pos]
                v = other.d[pos]
                row_entries_b[row_idx].append((col_idx, v))

        temp_row_vals = [0.0] * rows_a

        for j in range(cols_b):
            for i in range(rows_a):
                temp_row_vals[i] = 0.0

            for i in range(rows_b):
                for cb, vb in row_entries_b[i]:
                    if cb == j:
                        col_st = self.ptrs[i]
                        col_en = self.ptrs[i + 1]

                        for a_pos in range(col_st, col_en):
                            ra = self.idxs[a_pos]
                            va = self.d[a_pos]
                            temp_row_vals[ra] += va * vb

            for i in range(rows_a):
                if abs(temp_row_vals[i]) > 1e-14:
                    res_d.append(temp_row_vals[i])
                    res_idxs.append(i)

            res_ptrs[j + 1] = len(res_d)

        return CSCMatrix(res_d, res_idxs, res_ptrs, (rows_a, cols_b))

    @classmethod
    def from_dense(cls, dense_mat: DenseMatrix) -> 'CSCMatrix':
        rows_cnt = len(dense_mat)
        cols_cnt = len(dense_mat[0])
        
        vals = []
        idxs = []
        
        cnts_per_col = [0] * cols_cnt
        
        for j in range(cols_cnt):
            for i in range(rows_cnt):
                v = dense_mat[i][j]
                if v != 0:
                    vals.append(v)
                    idxs.append(i)
                    cnts_per_col[j] += 1
        
        ptrs = [0] * (cols_cnt + 1)
        for j in range(cols_cnt):
            ptrs[j + 1] = ptrs[j] + cnts_per_col[j]
        
        return CSCMatrix(vals, idxs, ptrs, (rows_cnt, cols_cnt))

    def _to_csr(self) -> 'CSRMatrix':
        from CSR import CSRMatrix
    
        m_val, n_val = self.s

        cnts_per_row = [0] * m_val
        for row_idx in self.idxs:
            cnts_per_row[row_idx] += 1

        new_ptrs = [0] * (m_val + 1)
        for i in range(m_val):
            new_ptrs[i + 1] = new_ptrs[i] + cnts_per_row[i]

        new_vals = [0] * len(self.d)
        new_idxs = [0] * len(self.idxs)

        cur_pos = new_ptrs.copy()

        for j in range(n_val):
            col_st = self.ptrs[j]
            col_en = self.ptrs[j + 1]

            for k in range(col_st, col_en):
                i_val = self.idxs[k]
                v_val = self.d[k]

                p_val = cur_pos[i_val]

                new_vals[p_val] = v_val
                new_idxs[p_val] = j

                cur_pos[i_val] += 1

        return CSRMatrix(new_vals, new_idxs, new_ptrs, (m_val, n_val))

    def _to_coo(self) -> 'COOMatrix':
        from COO import COOMatrix

        rows_cnt, cols_cnt = self.s

        vals = []
        row_idxs = []
        col_idxs = []
        
        for j in range(cols_cnt):
            st = self.ptrs[j]
            en = self.ptrs[j + 1]
            
            for pos in range(st, en):
                i_val = self.idxs[pos]
                v_val = self.d[pos]
                
                vals.append(v_val)
                row_idxs.append(i_val)
                col_idxs.append(j)
        
        return COOMatrix(vals, row_idxs, col_idxs, self.s)