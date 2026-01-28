from base import Matrix
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from COO import COOMatrix
    from CSR import CSRMatrix

class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)

        if len(indptr) != shape[1] + 1:
            raise ValueError(f"Неверная длина indptr")
        if indptr[0] != 0:
            raise ValueError("indptr[0] должен быть 0")
        if indptr[-1] != len(data):
            raise ValueError(f"Несоответствие indptr[-1]")
        if len(data) != len(indices):
            raise ValueError(f"Разные длины массивов")
        
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        r_cnt, c_cnt = self.shape

        res_mat = [[0] * c_cnt for _ in range(r_cnt)]
        
        for j in range(c_cnt):
            st = self.indptr[j]
            en = self.indptr[j + 1]

            for pos in range(st, en):
                i = self.indices[pos]
                v = self.data[pos]
                res_mat[i][j] = v
        
        return res_mat

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        coo_a = self._to_coo()
        coo_b = other._to_coo()
        coo_res = coo_a._add_impl(coo_b)
        
        return coo_res._to_csc()

    def _mul_impl(self, k: float) -> 'Matrix':
        if k == 0:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)
        
        new_vals = [x * k for x in self.data]
        
        return CSCMatrix(new_vals, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        from CSR import CSRMatrix

        r_cnt, c_cnt = self.shape
        nr_cnt, nc_cnt = c_cnt, r_cnt
        
        row_cnts = [0] * nr_cnt
        
        for j in range(c_cnt):
            st = self.indptr[j]
            en = self.indptr[j + 1]
            row_cnts[j] = en - st
        
        new_indptr = [0] * (nr_cnt + 1)
        for i in range(nr_cnt):
            new_indptr[i + 1] = new_indptr[i] + row_cnts[i]
        
        new_data = [0] * len(self.data)
        new_indices = [0] * len(self.indices)
        
        row_pos = new_indptr.copy()
        
        for j in range(c_cnt):
            st = self.indptr[j]
            en = self.indptr[j + 1]
            
            for pos in range(st, en):
                i = self.indices[pos]
                v = self.data[pos]
                
                p = row_pos[j]
                new_data[p] = v
                new_indices[p] = i
                row_pos[j] += 1
        
        return CSRMatrix(new_data, new_indices, new_indptr, (nr_cnt, nc_cnt))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        ra, ca = self.shape
        rb, cb = other.shape

        res_d = []
        res_idx = []
        res_ptr = [0] * (cb + 1)

        rb = other.shape[0]
        cb = other.shape[1]

        row_entries = [[] for _ in range(rb)]

        for col in range(cb):
            st = other.indptr[col]
            en = other.indptr[col + 1]
            for pos in range(st, en):
                row = other.indices[pos]
                v = other.data[pos]
                row_entries[row].append((col, v))

        temp_row = [0.0] * ra

        for j in range(cb):
            for i in range(ra):
                temp_row[i] = 0.0

            for i in range(rb):
                for cb_val, vb in row_entries[i]:
                    if cb_val == j:
                        col_st = self.indptr[i]
                        col_en = self.indptr[i + 1]

                        for a_pos in range(col_st, col_en):
                            ra_val = self.indices[a_pos]
                            va = self.data[a_pos]
                            temp_row[ra_val] += va * vb

            for i in range(ra):
                if abs(temp_row[i]) > 1e-14:
                    res_d.append(temp_row[i])
                    res_idx.append(i)

            res_ptr[j + 1] = len(res_d)

        return CSCMatrix(res_d, res_idx, res_ptr, (ra, cb))

    @classmethod
    def from_dense(cls, dense_mat: DenseMatrix) -> 'CSCMatrix':
        r_cnt = len(dense_mat)
        c_cnt = len(dense_mat[0])
        
        vals = []
        idxs = []
        
        col_cnts = [0] * c_cnt
        
        for j in range(c_cnt):
            for i in range(r_cnt):
                v = dense_mat[i][j]
                if v != 0:
                    vals.append(v)
                    idxs.append(i)
                    col_cnts[j] += 1
        
        ptrs = [0] * (c_cnt + 1)
        for j in range(c_cnt):
            ptrs[j + 1] = ptrs[j] + col_cnts[j]
        
        return CSCMatrix(vals, idxs, ptrs, (r_cnt, c_cnt))

    def _to_csr(self) -> 'CSRMatrix':
        from CSR import CSRMatrix
    
        m, n = self.shape

        row_cnts = [0] * m
        for row_idx in self.indices:
            row_cnts[row_idx] += 1

        new_ptr = [0] * (m + 1)
        for i in range(m):
            new_ptr[i + 1] = new_ptr[i] + row_cnts[i]

        new_vals = [0] * len(self.data)
        new_idxs = [0] * len(self.indices)

        cur_pos = new_ptr.copy()

        for j in range(n):
            col_st = self.indptr[j]
            col_en = self.indptr[j + 1]

            for k in range(col_st, col_en):
                i_val = self.indices[k]
                v_val = self.data[k]

                p_val = cur_pos[i_val]

                new_vals[p_val] = v_val
                new_idxs[p_val] = j

                cur_pos[i_val] += 1

        return CSRMatrix(new_vals, new_idxs, new_ptr, (m, n))

    def _to_coo(self) -> 'COOMatrix':
        from COO import COOMatrix

        r_cnt, c_cnt = self.shape

        vals = []
        rows = []
        cols = []
        
        for j in range(c_cnt):
            st = self.indptr[j]
            en = self.indptr[j + 1]
            
            for pos in range(st, en):
                i_val = self.indices[pos]
                v_val = self.data[pos]
                
                vals.append(v_val)
                rows.append(i_val)
                cols.append(j)
        
        return COOMatrix(vals, rows, cols, self.shape)