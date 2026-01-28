from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix
import sys

class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
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
        other_coo = other._to_coo()

        n1 = len(self.data)
        n2 = len(other_coo.data)

        all_row = [0] * (n1 + n2)
        all_col = [0] * (n1 + n2)
        all_val = [0.0] * (n1 + n2)

        for i in range(n1):
            all_row[i] = self.row[i]
            all_col[i] = self.col[i]
            all_val[i] = self.data[i]

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
        self_dense = self.to_dense()
        other_dense = other.to_dense()

        rows, cols = self.shape[0], other.shape[1]
        result_data, result_rows, result_cols = [], [], []

        for i in range(rows):
            for j in range(cols):
                sum_val = 0.0
                for k in range(self.shape[1]):
                    sum_val += self_dense[i][k] * other_dense[k][j]
                if abs(sum_val) > 1e-10:
                    result_rows.append(i)
                    result_cols.append(j)
                    result_data.append(sum_val)

        return COOMatrix(result_data, result_rows, result_cols, (rows, cols))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        r, c, v = [], [], []
        for i in range(rows):
            for j in range(cols):
                val = dense_matrix[i][j]
                if abs(val) > 1e-10:
                    r.append(i)
                    c.append(j)
                    v.append(val)
        return cls(v, r, c, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        entries = list(zip(self.col, self.row, range(len(self.row))))
        entries.sort()

        sorted_data = [self.data[i] for _, _, i in entries]
        sorted_rows = [r for _, r, _ in entries]
        indptr = [0] * (self.shape[1] + 1)
        for idx, (c, _, _) in enumerate(entries):
            indptr[c + 1] += 1
        for i in range(1, self.shape[1] + 1):
            indptr[i] += indptr[i - 1]

        CSCClass = getattr(sys.modules['CSC'], 'CSCMatrix')
        return CSCClass(sorted_data, sorted_rows, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        elements = []
        for i in range(len(self.row)):
            elements.append((self.col[i], self.row[i], self.data[i]))

        elements.sort()
        csc_data = []
        csc_rows = []
        for element in elements:
            csc_data.append(element[2])
            csc_rows.append(element[1])

        indptr = [0]
        cols_count = self.shape[1]
        current_col_nnz = 0
        prev_col = -1

        for element in elements:
            this_col = element[0]
            if this_col != prev_col:
                indptr.append(indptr[-1] + current_col_nnz)
                current_col_nnz = 0
                prev_col = this_col
            current_col_nnz += 1

        indptr.append(indptr[-1] + current_col_nnz)
        CSRClass = getattr(sys.modules['CSR'], 'CSRMatrix')
        return CSRClass(csc_data, csc_rows, indptr, self.shape)