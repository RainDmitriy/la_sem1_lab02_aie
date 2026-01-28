from base import Matrix
from math import fabs
from typing import List, Tuple

# Основные типы данных
DenseMatrix = List[List[float]]  # Плотная матрица: [[row1], [row2], ...] как в NumPy
Shape = Tuple[int, int]  # Размерность: (rows, cols)

# Для CSR и CSC
CSRData = CSCData = List[float]  # Ненулевые значения
CSRIndices = CSCIndices = List[int]  # Колонки (CSR) или строки (CSC)
CSRIndptr = CSCIndptr = List[int]  # Указатели начала строк (CSR) или колонок (CSC)


class CSCMatrix(Matrix):
    # Константа для сравнения с нулем
    ZERO_TOLERANCE = 1e-15

    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr

        # Проверка согласованности
        assert len(data) == len(indices), "Data and indices must have same length"
        assert len(indptr) == shape[1] + 1, f"indptr length must be shape[1] + 1 = {shape[1] + 1}"
        assert indptr[0] == 0, "indptr must start with 0"
        assert indptr[-1] == len(data), f"Last indptr must equal data length: {indptr[-1]} != {len(data)}"

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу."""
        n_rows, n_cols = self.shape
        dense_data = [[0.0 for _ in range(n_cols)] for _ in range(n_rows)]

        for col in range(n_cols):
            start = self.indptr[col]
            end = self.indptr[col + 1]
            for idx in range(start, end):
                row = self.indices[idx]
                value = self.data[idx]
                dense_data[row][col] = value

        return dense_data

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""

        assert isinstance(other, CSCMatrix), "Addition is only supported between two CSC matrices"
        assert self.shape == other.shape, "Matrix shapes must match for addition"

        n_rows, n_cols = self.shape
        result_data, result_indices, result_indptr = [], [], [0]

        for col in range(n_cols):
            self_start, self_end = self.indptr[col], self.indptr[col + 1]
            other_start, other_end = other.indptr[col], other.indptr[col + 1]

            i, j = self_start, other_start

            while i < self_end and j < other_end:
                row1, row2 = self.indices[i], other.indices[j]

                if row1 < row2:
                    # Элемент только в первой матрице
                    result_indices.append(row1)
                    result_data.append(self.data[i])
                    i += 1
                elif row1 > row2:
                    # Элемент только во второй матрице
                    result_indices.append(row2)
                    result_data.append(other.data[j])
                    j += 1
                else:
                    # Элемент в обеих матрицах
                    sum_val = self.data[i] + other.data[j]
                    if fabs(sum_val) >= self.ZERO_TOLERANCE:
                        result_indices.append(row1)
                        result_data.append(sum_val)
                    i += 1
                    j += 1

            # Остатки из первой матрицы
            while i < self_end:
                result_indices.append(self.indices[i])
                result_data.append(self.data[i])
                i += 1

            # Остатки из второй матрицы
            while j < other_end:
                result_indices.append(other.indices[j])
                result_data.append(other.data[j])
                j += 1

            result_indptr.append(len(result_data))

        return CSCMatrix(result_data, result_indices, result_indptr, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        if fabs(scalar) < self.ZERO_TOLERANCE:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)

        new_data = [val * scalar for val in self.data]

        return CSCMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Результат - в CSR формате (с теми же данными, но с интерпретацией строк как столбцов).
        """
        from CSR import CSRMatrix
        return CSRMatrix(self.data.copy(), self.indices.copy(), self.indptr.copy(), (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""

        assert self.shape[1] == other.shape[0], "Matrix dimensions incompatible for multiplication"

        if not isinstance(other, CSCMatrix):
            if hasattr(other, "_to_csc"):
                other_csc = other._to_csc()
            else:
                other_csc = CSCMatrix.from_dense(other.to_dense())
        else:
            other_csc = other

        n_rows, inner_dim = self.shape
        n_cols = other_csc.shape[1]

        self_csr = self._to_csr()

        result_dict = {}

        for i in range(n_rows):
            row_start = self_csr.indptr[i]
            row_end = self_csr.indptr[i + 1]

            for p1 in range(row_start, row_end):
                k = self_csr.indices[p1]
                val_a = self_csr.data[p1]

                col_start = other_csc.indptr[k]
                col_end = other_csc.indptr[k + 1]

                for p2 in range(col_start, col_end):
                    j = other_csc.indices[p2]
                    val_b = other_csc.data[p2]

                    key = (i, j)
                    result_dict[key] = result_dict.get(key, 0.0) + val_a * val_b

        if not result_dict:
            return CSCMatrix([], [], [0] * (n_cols + 1), (n_rows, n_cols))

        cols_dict = {}
        for (i, j), val in result_dict.items():
            if abs(val) >= self.ZERO_TOLERANCE:
                if j not in cols_dict:
                    cols_dict[j] = []
                cols_dict[j].append((i, val))

        data = []
        indices = []
        indptr = [0]

        current_pos = 0
        for col_idx in range(n_cols):
            if col_idx in cols_dict:
                elements = cols_dict[col_idx]
                elements.sort(key=lambda x: x[0])

                for row, val in elements:
                    data.append(val)
                    indices.append(row)

                current_pos += len(elements)

            indptr.append(current_pos)

        return CSCMatrix(data, indices, indptr, (n_rows, n_cols))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        n_rows, n_cols = len(dense_matrix), len(dense_matrix[0])

        data, indices, indptr = [], [], [0]

        for col in range(n_cols):
            for row in range(n_rows):
                val = dense_matrix[row][col]
                if fabs(val) >= cls.ZERO_TOLERANCE:
                    data.append(val)
                    indices.append(row)
            indptr.append(len(data))

        return cls(data, indices, indptr, (n_rows, n_cols))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix.
        """

        n_rows, n_cols = self.shape

        row_counts = [0] * n_rows

        for row_idx in self.indices:
            row_counts[row_idx] += 1

        row_ptr = [0] * (n_rows + 1)
        for i in range(n_rows):
            row_ptr[i + 1] = row_ptr[i] + row_counts[i]

        positions = row_ptr.copy()
        new_data = [0.0] * len(self.data)
        new_indices = [0] * len(self.indices)

        for col in range(n_cols):
            start, end = self.indptr[col], self.indptr[col + 1]
            for idx in range(start, end):
                row = self.indices[idx]
                new_row = col
                new_col = row

                pos = positions[new_row]
                new_data[pos] = self.data[idx]
                new_indices[pos] = new_col
                positions[new_row] += 1

        from CSR import CSRMatrix
        return CSRMatrix(new_data, new_indices, row_ptr, (n_rows, n_cols))

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        data, rows, cols = [], [], []

        for col in range(self.shape[1]):
            start, end = self.indptr[col], self.indptr[col + 1]
            for idx in range(start, end):
                data.append(self.data[idx])
                rows.append(self.indices[idx])
                cols.append(col)

        from COO import COOMatrix
        return COOMatrix(data, rows, cols, self.shape)
