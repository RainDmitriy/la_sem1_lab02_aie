from base import Matrix
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from CSR import CSRMatrix
    from CSC import CSCMatrix
    from COO import COOMatrix

class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        # if len(indptr) != shape[1] + 1:
        #     raise ValueError(f"indptr должен иметь длину {shape[1] + 1}")
        # if indptr[0] != 0 or indptr[-1] != len(data):
        #     raise ValueError("Некорректный indptr")
        # if len(data) != len(indices):
        #     raise ValueError("Длины data и indices должны совпадать")
        self.data, self.indices, self.indptr  = data, indices, indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0 for i in range(cols)] for i in range(rows)]
        for j in range(cols):
            for i in range(self.indptr[j], self.indptr[j + 1]):
                dense[self.indices[i]][j] = self.data[i]
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        self_coo = self._to_coo()
        other_coo = other._to_coo()
        result_coo = self_coo._add_impl(other_coo)
        return result_coo._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        n = len(self.data)
        new_data = [0.0] * n
        for i in range(n):
            new_data[i] = self.data[i] * scalar
        return CSCMatrix(new_data, self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Hint:
        Результат - в CSR формате (с теми же данными, но с интерпретацией строк как столбцов).
        """
        new_row = []
        new_col = []
        new_data = []
        for j in range(self.shape[1]):
            col_start = self.indptr[j]
            col_end = self.indptr[j + 1]
            for k in range(col_start, col_end):
                new_row.append(j)
                new_col.append(self.indices[k])
                new_data.append(self.data[k])
        from COO import COOMatrix
        coo_result = COOMatrix(new_data, new_row, new_col, (self.shape[1], self.shape[0]))
        return coo_result._to_csr()

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        other_csr = other._to_csr()
        rows_a = self.shape[0]
        cols_b = other_csr.shape[1]
        res_r, res_c, res_v = [], [], []
        for j in range(cols_b):
            for i in range(rows_a):
                s = 0.0
                row_start_a = self.indptr[i]
                row_end_a = self.indptr[i + 1]
                col_start_b = other_csr.indptr[j]
                col_end_b = other_csr.indptr[j + 1]
                k1 = row_start_a
                k2 = col_start_b
                while k1 < row_end_a and k2 < col_end_b:
                    if self.indices[k1] == other_csr.indices[k2]:
                        s += self.data[k1] * other_csr.data[k2]
                        k1 += 1
                        k2 += 1
                    elif self.indices[k1] < other_csr.indices[k2]:
                        k1 += 1
                    else:
                        k2 += 1
                if abs(s) > 1e-10:
                    res_r.append(i)
                    res_c.append(j)
                    res_v.append(s)
        from COO import COOMatrix
        coo_result = COOMatrix(res_v, res_r, res_c, (rows_a, cols_b))
        return coo_result._to_csc()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        csc_data, csc_indices, csc_indptr = [], [], [0]
        for j in range(cols):
            col_nz = 0
            for i in range(rows):
                val = dense_matrix[i][j]
                if abs(val) > 1e-10:
                    csc_indices.append(i)
                    csc_data.append(val)
                    col_nz += 1
            csc_indptr.append(csc_indptr[-1] + col_nz)
        return cls(csc_data, csc_indices, csc_indptr, (rows, cols))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix
        coo = self._to_coo()
        temp_rows = [[] for _ in range(self.shape[0])]
        for i in range(len(coo.data)):
            temp_rows[coo.row[i]].append((coo.col[i], coo.data[i]))
        csr_data, csr_indices, csr_indptr = [], [], [0]
        for r in range(self.shape[0]):
            temp_rows[r].sort()
            for c, val in temp_rows[r]:
                csr_indices.append(c)
                csr_data.append(val)
            csr_indptr.append(len(csr_data))
        return CSRMatrix(csr_data, csr_indices, csr_indptr, self.shape)

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        new_row = []
        new_col = []
        new_data = []
        for j in range(self.shape[1]):
            col_start = self.indptr[j]
            col_end = self.indptr[j + 1]
            for k in range(col_start, col_end):
                new_row.append(self.indices[k])
                new_col.append(j)
                new_data.append(self.data[k])
        from COO import COOMatrix
        return COOMatrix(new_data, new_row, new_col, self.shape)