from base import Matrix
from typing import List, Tuple

DenseMatrix = List[List[float]]  # Плотная матрица: [[row1], [row2], ...] как в NumPy
Shape = Tuple[int, int]  # Размерность: (rows, cols)

# Для COO
COOData = List[float]      # Ненулевые значения
COORows = List[int]        # Индексы строк
COOCols = List[int]        # Индексы столбцов


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
        assert isinstance(other, COOMatrix), "Addition is only supported between two COO matrices"
        assert self.shape == other.shape, "Matrix shapes must match for addition"

        # Предполагаем, что обе матрицы уже отсортированы по (row, col)
        # В рамках моей реализации монотонность гарантированна

        result_data, result_row, result_col = [], [], []
        i, j = 0, 0
        n1, n2 = len(self.data), len(other.data)

        while i < n1 and j < n2:
            r1, c1 = self.row[i], self.col[i]
            r2, c2 = other.row[j], other.col[j]

            if (r1, c1) < (r2, c2):
                # Элемент только в первой матрице
                result_row.append(r1)
                result_col.append(c1)
                result_data.append(self.data[i])
                i += 1
            elif (r1, c1) > (r2, c2):
                # Элемент только во второй матрице
                result_row.append(r2)
                result_col.append(c2)
                result_data.append(other.data[j])
                j += 1
            else:
                # Элемент в обеих матрицах
                sum_val = self.data[i] + other.data[j]
                if abs(sum_val) >= self.ZERO_TOLERANCE:
                    result_row.append(r1)
                    result_col.append(c1)
                    result_data.append(sum_val)
                i += 1
                j += 1

        while i < n1:
            result_row.append(self.row[i])
            result_col.append(self.col[i])
            result_data.append(self.data[i])
            i += 1

        while j < n2:
            result_row.append(other.row[j])
            result_col.append(other.col[j])
            result_data.append(other.data[j])
            j += 1

        return COOMatrix(result_data, result_row, result_col, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        if abs(scalar) < self.ZERO_TOLERANCE:
            return COOMatrix([], [], [], self.shape)

        new_data = [val * scalar for val in self.data]

        data, row, col = [], [], []
        for val, r, c in zip(new_data, self.row, self.col):
            if abs(val) >= self.ZERO_TOLERANCE:
                data.append(val)
                row.append(r)
                col.append(c)

        return COOMatrix(data, row, col, self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        # Меняем местами строки и столбцы
        return COOMatrix(
            data=self.data.copy(),
            row=self.col.copy(),
            col=self.row.copy(),
            shape=(self.shape[1], self.shape[0])
        )

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

                if abs(dot_product) >= self.ZERO_TOLERANCE:
                    result_dict[(i, j)] = dot_product

        if not result_dict:
            return COOMatrix([], [], [], (self.shape[0], other.shape[1]))

        data, row, col = zip(*[(val, r, c) for (r, c), val in result_dict.items()])

        return COOMatrix(list(data), list(row), list(col), (self.shape[0], other.shape[1]))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        data, rows, cols = [], [], []
        dense_data = dense_matrix.data

        for i in range(len(dense_data)):
            for j in range(len(dense_data[0])):
                val = dense_data[i][j]
                if abs(val) >= 1e-10:
                    data.append(val)
                    rows.append(i)
                    cols.append(j)

        return cls(data, rows, cols, dense_matrix.shape)

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix

        # Сортируем по столбцам, затем по строкам
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

        return CSCMatrix(
            data=list(sorted_data),
            row_indices=list(sorted_rows),
            col_ptr=col_ptr,
            shape=self.shape
        )

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

        return CSRMatrix(
            data=list(sorted_data),
            col_indices=list(sorted_cols),
            row_ptr=row_ptr,
            shape=self.shape
        )
