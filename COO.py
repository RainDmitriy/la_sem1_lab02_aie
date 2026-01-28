from base import Matrix
from typing import List, Tuple

DenseMatrix = List[List[float]]  # Плотная матрица: [[row1], [row2], ...] как в NumPy
Shape = Tuple[int, int]  # Размерность: (rows, cols)

# Для COO
COOData = List[float]  # Ненулевые значения
COORows = List[int]  # Индексы строк
COOCols = List[int]  # Индексы столбцов


class COOMatrix(Matrix):
    ZERO_TOLERANCE = 1e-15

    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.row = row
        self.col = col
        self.shape = shape

        # Проверка корректности данных
        assert len(data) == len(row) == len(col), "Data, row and col must have same length"
        assert all(0 <= r < shape[0] for r in row), "Row indices out of bounds"
        assert all(0 <= c < shape[1] for c in col), "Col indices out of bounds"

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу."""
        n_rows, n_cols = self.shape
        dense_data = [[0.0 for _ in range(n_cols)] for _ in range(n_rows)]

        for val, r, c in zip(self.data, self.row, self.col):
            dense_data[r][c] = val

        return DenseMatrix(dense_data)

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц"""
        assert self.shape == other.shape, "Matrix shapes must match for addition"

        if hasattr(other, "_to_coo"):
            other_coo = other._to_coo()
        else:
            other_coo = COOMatrix.from_dense(other.to_dense())

        result_dict = {}

        for val, r, c in zip(self.data, self.row, self.col):
            result_dict[(r, c)] = result_dict.get((r, c), 0.0) + val

        for val, r, c in zip(other_coo.data, other_coo.row, other_coo.col):
            result_dict[(r, c)] = result_dict.get((r, c), 0.0) + val

        result_data, result_row, result_col = [], [], []
        for (r, c), val in result_dict.items():
            if abs(val) >= self.ZERO_TOLERANCE:
                result_data.append(val)
                result_row.append(r)
                result_col.append(c)

        return COOMatrix(result_data, result_row, result_col, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        new_data = [val * scalar for val in self.data]

        return COOMatrix(new_data, self.row, self.col, self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""

        return COOMatrix(self.data.copy(), self.col.copy(), self.row.copy(),(self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""

        assert isinstance(other, COOMatrix), "Multiplication is only supported between two COO matrices"
        assert self.shape[1] == other.shape[0], "Matrix dimensions incompatible for multiplication"

        result_dict = {}  # (i, j) -> значение

        rows_dict = {}
        for val, r, c in zip(self.data, self.row, self.col):
            if r not in rows_dict:
                rows_dict[r] = []
            rows_dict[r].append((c, val))

        cols_dict = {}
        for val, r, c in zip(other.data, other.row, other.col):
            if c not in cols_dict:
                cols_dict[c] = []
            cols_dict[c].append((r, val))

        for i in rows_dict:
            for j in cols_dict:
                dot_product = 0.0

                row_elems = rows_dict[i]
                col_elems = cols_dict[j]

                row_dict = {k: v for k, v in row_elems}
                for k, val2 in col_elems:
                    if k in row_dict:
                        dot_product += row_dict[k] * val2

                result_dict[(i, j)] = dot_product

        if not result_dict:
            return COOMatrix([], [], [], (self.shape[0], other.shape[1]))

        data, row, col = zip(*[(val, r, c) for (r, c), val in result_dict.items()])

        return COOMatrix(list(data), list(row), list(col), (self.shape[0], other.shape[1]))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        data, rows, cols = [], [], []

        for i in range(len(dense_matrix)):
            for j in range(len(dense_matrix[0])):
                val = dense_matrix[i][j]
                if abs(val) > 0.0:
                    data.append(val)
                    rows.append(i)
                    cols.append(j)

        return cls(data, rows, cols, (len(dense_matrix), len(dense_matrix[0])))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix

        sorted_indices = sorted(zip(self.col, self.row, self.data))

        if not sorted_indices:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)

        sorted_cols, sorted_rows, sorted_data = zip(*sorted_indices)

        n_cols = self.shape[1]
        col_ptr = [0] * (n_cols + 1)

        for col_idx in sorted_cols:
            col_ptr[col_idx + 1] += 1

        # Преобразуем в cumulative sum
        for i in range(1, n_cols + 1):
            col_ptr[i] += col_ptr[i - 1]

        return CSCMatrix(list(sorted_data), list(sorted_rows), col_ptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix
        # Сортируем по строкам, затем по столбцам (для CSR)
        sorted_indices = sorted(zip(self.row, self.col, self.data))

        if not sorted_indices:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)

        sorted_rows, sorted_cols, sorted_data = zip(*sorted_indices)

        n_rows = self.shape[0]
        row_ptr = [0] * (n_rows + 1)

        for row_idx in sorted_rows:
            row_ptr[row_idx + 1] += 1

        for i in range(1, n_rows + 1):
            row_ptr[i] += row_ptr[i - 1]

        return CSRMatrix(list(sorted_data), list(sorted_cols), row_ptr, self.shape)
