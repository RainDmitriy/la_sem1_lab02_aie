from base import Matrix
from .type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix

class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        pass
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC матрицу в плотную"""
        pass
        rows, cols = self.shape
        result_matrix = [[0] * cols for r in range(rows)]
        for c in range(cols):
            col_start = self.indptr[c]
            col_end = self.indptr[c + 1]
            for i in range(col_start, col_end):
                r = self.indices[i]
                val = self.data[i]
                if 0 <= r < rows:
                    result_matrix[r][c] = val
        return result_matrix

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц"""
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
        for c in range(cols):
            self_start = indptr_self[c]
            other_start = indptr_other[c]
            self_end = indptr_self[c + 1]
            other_end = indptr_other[c + 1]
            while self_start < self_end or other_start < other_end:
                if self_start == self_end:
                    r, val = indices_other[other_start], data_other[other_start]
                    other_start += 1
                elif other_start == other_end:
                    r, val = indices_self[self_start], data_self[self_start]
                    self_start += 1
                elif indices_self[self_start] < indices_other[other_start]:
                    r, val = indices_self[self_start], data_self[self_start]
                    self_start += 1
                elif indices_other[other_start] < indices_self[self_start]:
                    r, val = indices_other[other_start], data_other[other_start]
                    other_start += 1
                else:
                    r, val = indices_self[self_start], data_self[self_start] + data_other[other_start]
                    self_start += 1
                    other_start += 1
                result_indices.append(r)
                result_data.append(val)
            result_indptr.append(len(result_data))
        return CSCMatrix(result_data, result_indices, result_indptr, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр"""
        pass
        new_data = []
        for d in self.data:
            new_data.append(scalar * d)
        return CSCMatrix(new_data, self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы
        Результат в CSR формате
        """
        pass
        rows, cols = self.shape
        new_shape = (cols, rows)
        new_indptr = [0] * (rows + 1)
        row_counts = [0] * rows
        for i in range(len(self.indices)):
            r = self.indices[i]
            if 0 <= r < rows:
                row_counts[r] += 1
        for r in range(rows):
            new_indptr[r + 1] = new_indptr[r] + row_counts[r]
        new_indices = [0] * len(self.data)
        new_data = [0] * len(self.data)
        positions = list(new_indptr)
        for c in range(cols):
            col_start = self.indptr[c]
            col_end = self.indptr[c + 1] if c + 1 < len(self.indptr) else len(self.indices)
            for i in range(col_start, col_end):
                r = self.indices[i]
                val = self.data[i]
                if 0 <= r < rows:
                    pos = positions[r]
                    new_indices[pos] = c
                    new_data[pos] = val
                    positions[r] += 1
        return CSCMatrix(new_data, new_indices, new_indptr, new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц"""
        pass
        rows_self, cols_self = self.shape
        rows_other, cols_other = other.shape
        result_data = []
        result_indices = []
        result_indptr = [0]
        temp_values = [0] * rows_self
        marker = [-1] * rows_self
        for c in range(cols_other):
            col_rows = []
            other_start = other.indptr[c]
            other_end = other.indptr[c + 1]
            for j in range(other_start, other_end):
                k = other.indices[j]
                val_other = other.data[j]
                self_start = self.indptr[k]
                self_end = self.indptr[k + 1]
                for i in range(self_start, self_end):
                    r_self = self.indices[i]
                    val_self = self.data[i]
                    if marker[r_self] != c:
                        marker[r_self] = c
                        temp_values[r_self] = val_self * val_other
                        col_rows.append(r_self)
                    else:
                        temp_values[r_self] += val_self * val_other
            col_rows.sort()
            for r in col_rows:
                val = temp_values[r]
                if val != 0:
                    result_indices.append(r)
                    result_data.append(val)
            result_indptr.append(len(result_data))
        return CSCMatrix(result_data, result_indices, result_indptr, (rows_self, cols_other))


    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы"""
        pass
        matrix_row = len(dense_matrix)
        matrix_col = len(dense_matrix[0])
        csc_data = []
        csc_indices = []
        csc_indptr = [0]
        cnt = 0
        for c in range(matrix_col):
            for r in range(matrix_row):
                if dense_matrix[r][c] != 0:
                    cnt += 1
                    csc_data.append(dense_matrix[r][c])
                    csc_indices.append(r)
            csc_indptr.append(cnt)
        return cls(csc_data, csc_indices, csc_indptr, (matrix_row, matrix_col))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix
        """
        pass
        from CSR import CSRMatrix
        rows, cols = self.shape
        row_counts = [0] * rows
        for i in range(len(self.indices)):
            r = self.indices[i]
            if 0 <= r < rows:
                row_counts[r] += 1
        new_indptr = [0] * (rows + 1)
        for r in range(rows):
            new_indptr[r + 1] = new_indptr[r] + row_counts[r]
        new_indices = [0] * len(self.data)
        new_data = [0] * len(self.data)
        positions = list(new_indptr)
        for c in range(cols):
            col_start = self.indptr[c]
            col_end = self.indptr[c + 1] if c + 1 < len(self.indptr) else len(self.indices)
            for i in range(col_start, col_end):
                r = self.indices[i]
                val = self.data[i]
                if 0 <= r < rows:
                    pos = positions[r]
                    new_indices[pos] = c
                    new_data[pos] = val
                    positions[r] += 1
        return CSRMatrix(new_data, new_indices, new_indptr, self.shape)

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSCMatrix в COOMatrix
        """
        pass
        from COO import COOMatrix
        rows, cols = self.shape
        coo_data = list(self.data)
        coo_row = list(self.indices)
        cnt = len(self.data)
        coo_col = [0] * cnt

        for c in range(cols):
            start = self.indptr[c]
            end = self.indptr[c + 1]

            for i in range(start, end):
                coo_col[i] = c
        return COOMatrix(coo_data, coo_row, coo_col, self.shape)
