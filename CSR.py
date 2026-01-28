from base import Matrix
from .type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix

class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        pass
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR матрицу в плотную"""
        pass
        rows, cols = self.shape
        result_matrix = [[0] * cols for r in range(rows)]
        for r in range(rows):
            row_start = self.indptr[r]
            row_end = self.indptr[r + 1]
            for i in range(row_start, row_end):
                c = self.indices[i]
                val = self.data[i]
                if 0 <= c < cols:
                    result_matrix[r][c] = val
        return result_matrix

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матри"""
        pass
        data_self = list(self.data)
        indices_self = list(self.indices)
        indptr_self = list(self.indptr)
        data_other = list(other.data)
        indices_other = list(other.indices)
        indptr_other = list(other.indptr)
        result_data = []
        result_indices = []
        result_indptr = [0]
        rows, cols = self.shape
        for r in range(rows):
            self_start = indptr_self[r]
            other_start = indptr_other[r]
            self_end = indptr_self[r + 1]
            other_end = indptr_other[r + 1]
            while self_start < self_end or other_start < other_end:
                if self_start == self_end:
                    c, val = indices_other[other_start], data_other[other_start]
                    other_start += 1
                elif other_start == other_end:
                    c, val = indices_self[self_start], data_self[self_start]
                    self_start += 1
                elif indices_self[self_start] < indices_other[other_start]:
                    c, val = indices_self[self_start], data_self[self_start]
                    self_start += 1
                elif indices_other[other_start] < indices_self[self_start]:
                    c, val = indices_other[other_start], data_other[other_start]
                    other_start += 1
                else:
                    c, val = indices_self[self_start], data_self[self_start] + data_other[other_start]
                    self_start += 1
                    other_start += 1
                result_indices.append(c)
                result_data.append(val)
            result_indptr.append(len(result_data))
        return CSRMatrix(result_data, result_indices, result_indptr, self.shape)


    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр"""
        pass
        new_data = []
        for d in self.data:
            new_data.append(scalar * d)
        return CSRMatrix(new_data, self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы
        Hint:
        Результат в CSC формате
        """
        pass
        rows, cols = self.shape
        new_shape = (cols, rows)
        new_indptr = [0] * (cols + 1)
        col_counts = [0] * cols
        for i in range(len(self.indices)):
            c = self.indices[i]
            if 0 <= c < cols:
                col_counts[c] += 1
        for c in range(cols):
            new_indptr[c + 1] = new_indptr[c] + col_counts[c]
        new_indices = [0] * len(self.data)
        new_data = [0] * len(self.data)
        positions = list(new_indptr)
        for r in range(rows):
            row_start = self.indptr[r]
            row_end = self.indptr[r + 1] if r + 1 < len(self.indptr) else len(self.indices)
            for i in range(row_start, row_end):
                c = self.indices[i]
                if 0 <= c < cols:
                    pos = positions[c]
                    new_indices[pos] = r
                    new_data[pos] = self.data[i]
                    positions[c] += 1
        return CSRMatrix(new_data, new_indices, new_indptr, new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц"""
        pass
        rows_self, cols_self = self.shape
        rows_other, cols_other = other.shape
        result_data = []
        result_indices = []
        result_indptr = [0]
        temp_values = [0] * cols_other
        marker = [-1] * cols_other
        for r in range(rows_self):
            row_cols = []
            self_start = self.indptr[r]
            self_end = self.indptr[r + 1]
            for i in range(self_start, self_end):
                c_self = self.indices[i]
                val_self = self.data[i]
                other_start = other.indptr[c_self]
                other_end = other.indptr[c_self + 1]
                for j in range(other_start, other_end):
                    c_other = other.indices[j]
                    val_other = other.data[j]
                    if marker[c_other] != r:
                        marker[c_other] = r
                        temp_values[c_other] = val_self * val_other
                        row_cols.append(c_other)
                    else:
                        temp_values[c_other] += val_self * val_other
            row_cols.sort()
            for c in row_cols:
                val = temp_values[c]
                if val != 0:
                    result_indices.append(c)
                    result_data.append(val)
            result_indptr.append(len(result_data))
        return CSRMatrix(result_data, result_indices, result_indptr, (rows_self, cols_other))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR матрицы из плотной"""
        pass
        matrix_row = len(dense_matrix)
        matrix_col = len(dense_matrix[0])
        csr_data = []
        csr_indices = []
        csr_indptr = [0]
        cnt = 0
        for r in range(matrix_row):
            for c in range(matrix_col):
                if dense_matrix[r][c] != 0:
                    cnt += 1
                    csr_data.append(dense_matrix[r][c])
                    csr_indices.append(c)
            csr_indptr.append(cnt)
        return cls(csr_data, csr_indices, csr_indptr, (matrix_row, matrix_col))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSRMatrix в CSCMatrix
        """
        pass
        from CSC import CSCMatrix
        rows, cols = self.shape
        col_counts = [0] * cols
        for i in range(len(self.indices)):
            c = self.indices[i]
            if 0 <= c < cols:
                col_counts[c] += 1
        new_indptr = [0] * (cols + 1)
        for c in range(cols):
            new_indptr[c + 1] = new_indptr[c] + col_counts[c]
        cnt = len(self.data)
        new_indices = [0] * cnt
        new_data = [0] * cnt
        positions = list(new_indptr)
        for r in range(rows):
            row_start = self.indptr[r]
            row_end = self.indptr[r + 1]
            for i in range(row_start, row_end):
                c = self.indices[i]
                val = self.data[i]
                pos = positions[c]
                new_indices[pos] = r
                new_data[pos] = val
                positions[c] += 1
        return CSCMatrix(new_data, new_indices, new_indptr, (rows, cols))

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSRMatrix в COOMatrix
        """
        pass
        from COO import COOMatrix
        rows, cols = self.shape
        coo_data = list(self.data)
        coo_col = list(self.indices)
        cnt = len(self.data)
        coo_row = [0] * cnt
        for r in range(rows):
            start = self.indptr[r]
            end = self.indptr[r + 1]
            for i in range(start, end):
                coo_row[i] = r
        return COOMatrix(coo_data, coo_row, coo_col, self.shape)