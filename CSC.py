from base import Matrix
from types import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix
from CSR import CSRMatrix
from COO import COOMatrix

class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
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

        return COOMatrix(result_data, result_row, result_col, self.shape)

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
        n = self.shape[1]
        m = self.shape[0]

        new_row = []
        new_col = []
        new_data = []

        for j in range(n):
            col_start = self.indptr[j]
            col_end = self.indptr[j + 1]
            for k in range(col_start, col_end):
                new_row.append(j)
                new_col.append(self.indices[k])
                new_data.append(self.data[k])

        return COOMatrix(new_data, new_row, new_col, (n, m))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        if not isinstance(other, CSRMatrix):
            other_csr = other._to_csr()
        else:
            other_csr = other

        result_rows, result_cols, result_data = [], [], []
        rows_a, cols_a = other_csr.shape
        cols_b = self.shape[1]

        for j in range(cols_b):
            col_start = self.indptr[j]
            col_end = self.indptr[j + 1]

            for i in range(rows_a):
                row_start = other_csr.indptr[i]
                row_end = other_csr.indptr[i + 1]

                sum_val = 0.0
                k1 = col_start
                k2 = row_start

                while k1 < col_end and k2 < row_end:
                    if self.indices[k1] == other_csr.indices[k2]:
                        sum_val += self.data[k1] * other_csr.data[k2]
                        k1 += 1
                        k2 += 1
                    elif self.indices[k1] < other_csr.indices[k2]:
                        k1 += 1
                    else:
                        k2 += 1

                if abs(sum_val) > 1e-10:
                    result_rows.append(i)
                    result_cols.append(j)
                    result_data.append(sum_val)

        return COOMatrix(result_data, result_rows, result_cols, (rows_a, cols_b))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        rows = len(dense_matrix)
        if rows == 0:
            cols = 0
        else:
            cols = len(dense_matrix[0])

        data, indices, indptr = [], [], [0]

        for j in range(cols):
            col_nnz = 0
            for i in range(rows):
                val = dense_matrix[i][j]
                if abs(val) > 1e-10:
                    data.append(val)
                    indices.append(i)
                    col_nnz += 1
            indptr.append(indptr[-1] + col_nnz)

        return cls(data, indices, indptr, (rows, cols))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        coo = self._to_coo()
        return coo._to_csr()

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        row_list, col_list, data_list = [], [], []
        cols = self.shape[1]

        for j in range(cols):
            col_start = self.indptr[j]
            col_end = self.indptr[j + 1]
            for k in range(col_start, col_end):
                row_list.append(self.indices[k])
                col_list.append(j)
                data_list.append(self.data[k])

        return COOMatrix(data_list, row_list, col_list, self.shape)