from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix
import sys

class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data, self.indices, self.indptr  = data, indices, indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0 for i in range(cols)] for i in range(rows)]
        for i in range(rows):
            for j in range(self.indptr[i], self.indptr[i + 1]):
                dense[i][self.indices[j]] = self.data[j]
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        other_coo = other._to_coo()
        self_coo = self._to_coo()

        n1 = len(self_coo.data)
        n2 = len(other_coo.data)

        all_row = [0] * (n1 + n2)
        all_col = [0] * (n1 + n2)
        all_val = [0.0] * (n1 + n2)

        for i in range(n1):
            all_row[i] = self_coo.row[i]
            all_col[i] = self_coo.col[i]
            all_val[i] = self_coo.data[i]

        for i in range(n2):
            all_row[n1 + i] = other_coo.row[i]
            all_col[n1 + i] = other_coo.col[i]
            all_val[n1 + i] = other_coo.data[i]

        elements = []
        for i in range(n1 + n2):
            elements.append((all_row[i], all_col[i], all_val[i]))
        elements.sort()

        result_row, result_col, result_data = [], [], []
        i = 0
        total = n1 + n2

        while i < total:
            curr_row, curr_col, curr_val = elements[i]
            sum_val = curr_val
            i += 1
            while i < total and elements[i][0] == curr_row and elements[i][1] == curr_col:
                sum_val += elements[i][2]
                i += 1
            if abs(sum_val) > 1e-10:
                result_row.append(curr_row)
                result_col.append(curr_col)
                result_data.append(sum_val)

        COOClass = getattr(sys.modules['COO'], 'COOMatrix')
        coo_result = COOClass(result_data, result_row, result_col, self.shape)
        return coo_result._to_csr()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        n = len(self.data)
        new_data = [0.0] * n
        for i in range(n):
            new_data[i] = self.data[i] * scalar
        return CSRMatrix(new_data, self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Hint:
        Результат - в CSC формате (с теми же данными, но с интерпретацией столбцов как строк).
        """
        rows = self.shape[0]
        cols = self.shape[1]

        new_row = []
        new_col = []
        new_data = []

        for i in range(rows):
            row_start = self.indptr[i]
            row_end = self.indptr[i + 1]
            for k in range(row_start, row_end):
                new_row.append(self.indices[k])
                new_col.append(i)
                new_data.append(self.data[k])

        COOClass = getattr(sys.modules['COO'], 'COOMatrix')
        coo_result = COOClass(new_data, new_row, new_col, (cols, rows))
        return coo_result._to_csc()

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        other_csc = other._to_csc()

        rows = self.shape[0]
        cols = other_csc.shape[1]

        res_r, res_c, res_v = [], [], []

        for i in range(rows):
            row_start = self.indptr[i]
            row_end = self.indptr[i + 1]

            for j in range(cols):
                col_start = other_csc.indptr[j]
                col_end = other_csc.indptr[j + 1]

                s = 0.0
                k1 = row_start
                k2 = col_start

                while k1 < row_end and k2 < col_end:
                    if self.indices[k1] == other_csc.indices[k2]:
                        s += self.data[k1] * other_csc.data[k2]
                        k1 += 1
                        k2 += 1
                    elif self.indices[k1] < other_csc.indices[k2]:
                        k1 += 1
                    else:
                        k2 += 1

                if abs(s) > 1e-10:
                    res_r.append(i)
                    res_c.append(j)
                    res_v.append(s)

        COOClass = getattr(sys.modules['COO'], 'COOMatrix')
        return COOClass(res_v, res_r, res_c, (rows, cols))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        csr_data, csr_indices, csr_indptr = [], [], [0]
        for i in range(rows):
            row_nz = 0
            for j in range(cols):
                val = dense_matrix[i][j]
                if abs(val) > 1e-10:
                    csr_indices.append(j)
                    csr_data.append(val)
                    row_nz += 1
            csr_indptr.append(csr_indptr[-1] + row_nz)
        return cls(csr_data, csr_indices, csr_indptr, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSRMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix
        coo = self._to_coo()
        temp_cols = [[] for _ in range(self.shape[1])]
        for i in range(len(coo.data)):
            temp_cols[coo.col[i]].append((coo.row[i], coo.data[i]))
        csc_data, csc_indices, csc_indptr = [], [], [0]
        for c in range(self.shape[1]):
            temp_cols[c].sort()
            for r, val in temp_cols[c]:
                csc_indices.append(r)
                csc_data.append(val)
            csc_indptr.append(len(csc_data))
        return CSCMatrix(csc_data, csc_indices, csc_indptr, self.shape)
    
    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSRMatrix в COOMatrix.
        """
        n = self.shape[1]
        m = self.shape[0]

        new_row = []
        new_col = []
        new_data = []

        for i in range(m):
            row_start = self.indptr[i]
            row_end = self.indptr[i + 1]
            for k in range(row_start, row_end):
                new_row.append(self.indices[k])
                new_col.append(i)
                new_data.append(self.data[k])

        COOClass = getattr(sys.modules['COO'], 'COOMatrix')
        return COOClass(new_data, new_row, new_col, (n, m))