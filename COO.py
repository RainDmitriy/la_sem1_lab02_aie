from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from CSR import CSRMatrix
    from CSC import CSCMatrix
    from COO import COOMatrix

class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        if len(data) != len(row) or len(data) != len(col):
            raise ValueError("Длины data, row, col должны совпадать")
        self.data, self.row, self.col = data, row, col

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0 for i in range(cols)] for i in range(rows)]
        n = len(self.data)
        for k in range(n):
            dense[self.row[k]][self.col[k]] = self.data[k]
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц."""
        sum_dict = {}

        # добавляю элементы self
        for i in range(len(self.data)):
            key = (self.row[i], self.col[i])
            sum_dict[key] = sum_dict.get(key, 0.0) + self.data[i]
        other_coo = other._to_coo()
        for i in range(len(other_coo.data)):
            key = (other_coo.row[i], other_coo.col[i])
            sum_dict[key] = sum_dict.get(key, 0.0) + other_coo.data[i]
        # сортирую по (row, col) и фильтрую нули
        sorted_items = sorted(sum_dict.items(), key=lambda x: (x[0][0], x[0][1]))
        result_row, result_col, result_data = [], [], []
        for (r, c), val in sorted_items:
            if abs(val) > 1e-10:
                result_row.append(r)
                result_col.append(c)
                result_data.append(val)

        return COOMatrix(result_data, result_row, result_col, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        n = len(self.data)
        new_d = [0.0] * n
        for i in range(n):
            new_d[i] = self.data[i] * scalar
        return COOMatrix(new_d, self.row[:], self.col[:], self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        n = len(self.data)
        new_row = [0] * n
        new_col = [0] * n
        for i in range(n):
            new_row[i] = self.col[i]
            new_col[i] = self.row[i]
        return COOMatrix(self.data[:], new_row, new_col, (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""
        other_csr = other._to_csr()
        rows, cols = self.shape[0], other_csr.shape[1]
        result_data, result_row, result_col = [], [], []
        for i in range(len(self.data)):
            r_i = self.row[i]
            c_i = self.col[i]
            val_i = self.data[i]
            row_start = other_csr.indptr[c_i]
            row_end = other_csr.indptr[c_i + 1]
            for k in range(row_start, row_end):
                c_j = other_csr.indices[k]
                val_j = other_csr.data[k]
                val = val_i * val_j
                if abs(val) > 1e-10:
                    result_data.append(val)
                    result_row.append(r_i)
                    result_col.append(c_j)
        sum_dict = {}
        for i in range(len(result_data)):
            key = (result_row[i], result_col[i])
            sum_dict[key] = sum_dict.get(key, 0.0) + result_data[i]
        sorted_items = sorted(sum_dict.items(), key=lambda x: (x[0][0], x[0][1]))
        final_row, final_col, final_data = [], [], []
        for (r, c), val in sorted_items:
            final_row.append(r)
            final_col.append(c)
            final_data.append(val)
        return COOMatrix(final_data, final_row, final_col, (rows, cols))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        row_list, col_list, data_list = [], [], []
        for i in range(rows):
            for j in range(cols):
                val = dense_matrix[i][j]
                if abs(val) > 1e-10:
                    row_list.append(i)
                    col_list.append(j)
                    data_list.append(val)
        return cls(data_list, row_list, col_list, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix
        temp_cols = [[] for _ in range(self.shape[1])]
        for i in range(len(self.data)):
            temp_cols[self.col[i]].append((self.row[i], self.data[i]))

        csc_data, csc_indices, csc_indptr = [], [], [0]
        for c in range(self.shape[1]):
            temp_cols[c].sort()
            for r, val in temp_cols[c]:
                csc_indices.append(r)
                csc_data.append(val)
            csc_indptr.append(len(csc_data))
        return CSCMatrix(csc_data, csc_indices, csc_indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix
        temp_rows = [[] for _ in range(self.shape[0])]
        for i in range(len(self.data)):
            temp_rows[self.row[i]].append((self.col[i], self.data[i]))

        csr_data, csr_indices, csr_indptr = [], [], [0]
        for r in range(self.shape[0]):
            temp_rows[r].sort()
            for c, val in temp_rows[r]:
                csr_indices.append(c)
                csr_data.append(val)
            csr_indptr.append(len(csr_data))
        return CSRMatrix(csr_data, csr_indices, csr_indptr, self.shape)

    def _to_coo(self) -> 'COOMatrix':
        """COO возвращает сам себя."""
        return self