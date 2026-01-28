from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix

class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.row = row
        self.col = col

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO матрицу в плотную"""
        rows, cols = self.shape
        row_copy = list(self.row)
        col_copy = list(self.col)
        data_copy = list(self.data)
        result_matrix = [[0] * cols for r in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if ((r in row_copy) and (c in col_copy)) and row_copy.index(r) == col_copy.index(c):
                    result_matrix[r][c] = data_copy[row_copy.index(r)]
                    data_copy.pop(row_copy.index(r))
                    row_copy.pop(row_copy.index(r))
                    col_copy.pop(col_copy.index(c))
        return result_matrix

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO"""
        rows, cols = self.shape
        result_rows = []
        result_cols = []
        result_data = []
        row_copy = list(self.row)
        col_copy = list(self.col)
        data_copy = list(self.data)
        row_copy_other = list(other.row)
        col_copy_other = list(other.col)
        data_copy_other = list(other.data)
        for r in range(rows):
            for c in range(cols):
                self_index = None
                other_index = None
                for i in range(len(row_copy)):
                    if row_copy[i] == r and col_copy[i] == c:
                        self_index = i
                        break
                for i in range(len(row_copy_other)):
                    if row_copy_other[i] == r and col_copy_other[i] == c:
                        other_index = i
                        break
                if self_index is not None and other_index is not None:
                    result_rows.append(r)
                    result_cols.append(c)
                    result_data.append(data_copy[self_index] + data_copy_other[other_index])
                elif self_index is not None:
                    result_rows.append(r)
                    result_cols.append(c)
                    result_data.append(data_copy[self_index])
                elif other_index is not None:
                    result_rows.append(r)
                    result_cols.append(c)
                    result_data.append(data_copy_other[other_index])
        return COOMatrix(result_data, result_rows, result_cols, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр"""
        new_data = []
        for d in self.data:
            new_data.append(scalar * d)
        return COOMatrix(new_data, self.row, self.col, self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO"""
        rows, cols = self.shape
        new_shape = (cols, rows)
        new_row = list(self.col)
        new_col = list(self.row)
        return COOMatrix(self.data, new_row, new_col, new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц"""
        rows_self, cols_self = self.shape
        rows_other, cols_other = other.shape
        result_rows = []
        result_cols = []
        result_data = []
        row_copy_other = list(other.row)
        col_copy_other = list(other.col)
        data_copy_other = list(other.data)
        for row_self in range(rows_self):
            for col_other in range(cols_other):
                total = 0
                for col_self in range(cols_self):
                    self_val = 0
                    other_val = 0
                    for r in range(len(self.row)):
                        if self.row[r] == row_self and self.col[r] == col_self:
                            self_val = self.data[r]
                            break
                    for row_copy in range(len(row_copy_other)):
                        if row_copy_other[row_copy] == col_self and col_copy_other[row_copy] == col_other:
                            other_val = data_copy_other[row_copy]
                            break
                    total += self_val * other_val
                if total != 0:
                    result_rows.append(row_self)
                    result_cols.append(col_other)
                    result_data.append(total)
        return COOMatrix(result_data, result_rows, result_cols, (rows_self, cols_other))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO матрицы из плотной"""
        matrix_row = len(dense_matrix)
        matrix_col = len(dense_matrix[0])
        result_rows = []
        result_cols = []
        result_data = []
        for r in range(matrix_row):
            for c in range(matrix_col):
                if dense_matrix[r][c] != 0:
                    result_rows.append(r)
                    result_cols.append(c)
                    result_data.append(dense_matrix[r][c])
        return COOMatrix(result_data, result_rows, result_cols, (matrix_row, matrix_col))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix
        """
        from CSC import CSCMatrix
        rows, cols = self.shape
        col_counts = [0] * cols
        csc_data = [0] * len(self.data)
        csc_indices = [0] * len(self.row)
        csc_indptr = [0]
        positions = [0] * cols
        cnt = 0
        for c in self.col:
            col_counts[c] += 1
        for c in range(cols):
            cnt += col_counts[c]
            csc_indptr.append(cnt)
        for i in range(len(self.col)):
            c = self.col[i]
            pos = csc_indptr[c] + positions[c]
            csc_data[pos] = self.data[i]
            csc_indices[pos] = self.row[i]
            positions[c] += 1
        return CSCMatrix(csc_data, csc_indices, csc_indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix
        """
        from CSR import CSRMatrix
        rows, cols = self.shape
        row_counts = [0] * rows
        csr_data = [0] * len(self.data)
        csr_indices = [0] * len(self.col)
        temp_rows = [[] for _ in range(rows)]
        for i in range(len(self.data)):
            r = self.row[i]
            c = self.col[i]
            val = self.data[i]
            temp_rows[r].append((c, val))
        csr_data = []
        csr_indices = []
        csr_indptr = [0]
        positions = [0] * rows
        cnt = 0
        for r in self.row:
            row_counts[r] += 1
        for r in range(rows):
            cnt += row_counts[r]
            csr_indptr.append(cnt)
            temp_rows[r].sort()
            for c, val in temp_rows[r]:
                csr_indices.append(c)
                csr_data.append(val)
        for i in range(len(self.row)):
            r = self.row[i]
            pos = csr_indptr[r] + positions[r]
            csr_data[pos] = self.data[i]
            csr_indices[pos] = self.col[i]
            positions[r] += 1
            csr_indptr.append(len(csr_data))
        return CSRMatrix(csr_data, csr_indices, csr_indptr, self.shape)
