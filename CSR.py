from base import Matrix
from math import fabs
from typing import List, Tuple

# Основные типы данных
DenseMatrix = List[List[float]]  # Плотная матрица: [[row1], [row2], ...] как в NumPy
Shape = Tuple[int, int]  # Размерность: (rows, cols)

CSRData = CSCData = List[float]  # Ненулевые значения
CSRIndices = CSCIndices = List[int]  # Колонки (CSR) или строки (CSC)
CSRIndptr = CSCIndptr = List[int]  # Указатели начала строк (CSR) или колонок (CSC)


class CSRMatrix(Matrix):
    ZERO_TOLERANCE = 1e-15

    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr

        assert len(data) == len(indices), "Data and indices must have same length"
        assert len(indptr) == shape[0] + 1, f"indptr length must be shape[0] + 1 = {shape[0] + 1}"
        assert indptr[0] == 0, "indptr must start with 0"
        assert indptr[-1] == len(data), f"Last indptr must equal data length: {indptr[-1]} != {len(data)}"

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        n_rows, n_cols = self.shape
        dense_data = [[0.0 for _ in range(n_cols)] for _ in range(n_rows)]

        for row in range(n_rows):
            start = self.indptr[row]
            end = self.indptr[row + 1]
            for idx in range(start, end):
                col = self.indices[idx]
                value = self.data[idx]
                dense_data[row][col] = value

        return dense_data

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""

        assert isinstance(other, CSRMatrix), "Addition is only supported between two CSR matrices"
        assert self.shape == other.shape, "Matrix shapes must match for addition"

        n_rows, n_cols = self.shape
        result_data, result_indices, result_indptr = [], [], [0]

        for row in range(n_rows):
            self_start, self_end = self.indptr[row], self.indptr[row + 1]
            other_start, other_end = other.indptr[row], other.indptr[row + 1]

            i, j = self_start, other_start

            while i < self_end and j < other_end:
                col1, col2 = self.indices[i], other.indices[j]

                if col1 < col2:
                    # Элемент только в первой матрице
                    result_indices.append(col1)
                    result_data.append(self.data[i])
                    i += 1
                elif col1 > col2:
                    # Элемент только во второй матрице
                    result_indices.append(col2)
                    result_data.append(other.data[j])
                    j += 1
                else:
                    # Элемент в обеих матрицах
                    sum_val = self.data[i] + other.data[j]
                    if fabs(sum_val) >= self.ZERO_TOLERANCE:
                        result_indices.append(col1)
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

        return CSRMatrix(result_data, result_indices, result_indptr, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        if fabs(scalar) < self.ZERO_TOLERANCE:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)

        new_data = [val * scalar for val in self.data]

        return CSRMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Результат - в CSC формате (с теми же данными, но с интерпретацией столбцов как строк).
        """
        from CSC import CSCMatrix
        return CSCMatrix(self.data.copy(), self.indices.copy(), self.indptr.copy(), (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц"""
        from CSC import CSCMatrix

        assert self.shape[1] == other.shape[0], "Matrix dimensions incompatible for multiplication"

        if hasattr(other, "_to_csc"):
            other_csc = other._to_csc()
        else:
            other_csc = CSCMatrix.from_dense(other.to_dense())

        n_rows, inner_dim = self.shape
        n_cols = other_csc.shape[1]

        result_dict = {}

        for i in range(n_rows):
            row_start = self.indptr[i]
            row_end = self.indptr[i + 1]

            for p1 in range(row_start, row_end):
                k = self.indices[p1]
                val_a = self.data[p1]

                col_start = other_csc.indptr[k]
                col_end = other_csc.indptr[k + 1]

                for p2 in range(col_start, col_end):
                    j = other_csc.indices[p2]
                    val_b = other_csc.data[p2]

                    key = (i, j)
                    result_dict[key] = result_dict.get(key, 0.0) + val_a * val_b

        if not result_dict:
            return CSRMatrix([], [], [0] * (n_rows + 1), (n_rows, n_cols))

        rows_dict = {}
        for (i, j), val in result_dict.items():
            if abs(val) >= self.ZERO_TOLERANCE:
                if i not in rows_dict:
                    rows_dict[i] = []
                rows_dict[i].append((j, val))

        data = []
        indices = []
        indptr = [0]

        for row_idx in range(n_rows):
            if row_idx in rows_dict:
                elements = rows_dict[row_idx]
                elements.sort(key=lambda x: x[0])

                for col, val in elements:
                    data.append(val)
                    indices.append(col)

            indptr.append(len(data))

        return CSRMatrix(data, indices, indptr, (n_rows, n_cols))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        n_rows, n_cols = len(dense_matrix), len(dense_matrix[0])

        data, indices, indptr = [], [], [0]

        for row in range(n_rows):
            for col in range(n_cols):
                val = dense_matrix[row][col]
                if fabs(val) >= cls.ZERO_TOLERANCE:
                    data.append(val)
                    indices.append(col)
            indptr.append(len(data))

        return cls(data, indices, indptr, (n_rows, n_cols))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSRMatrix в CSCMatrix.
        """
        n_rows, n_cols = self.shape
        new_n_rows, new_n_cols = n_cols, n_rows

        col_counts = [0] * new_n_rows

        for col_idx in self.indices:
            col_counts[col_idx] += 1

        col_ptr = [0] * (new_n_rows + 1)
        for i in range(new_n_rows):
            col_ptr[i + 1] = col_ptr[i] + col_counts[i]

        positions = col_ptr.copy()
        new_data = [0.0] * len(self.data)
        new_indices = [0] * len(self.data)

        for row in range(n_rows):
            start, end = self.indptr[row], self.indptr[row + 1]
            for idx in range(start, end):
                col = self.indices[idx]
                new_col = row
                new_row = col

                pos = positions[new_row]
                new_data[pos] = self.data[idx]
                new_indices[pos] = new_col
                positions[new_row] += 1

        from CSC import CSCMatrix
        return CSCMatrix(new_data, new_indices, col_ptr, (new_n_rows, new_n_cols))

        return csc

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSRMatrix в COOMatrix.
        """
        data, rows, cols = [], [], []

        for row in range(self.shape[0]):
            start, end = self.indptr[row], self.indptr[row + 1]
            for idx in range(start, end):
                data.append(self.data[idx])
                rows.append(row)
                cols.append(self.indices[idx])

        from COO import COOMatrix
        return COOMatrix(data, rows, cols, self.shape)

    def __repr__(self) -> str:
        return f"CSRMatrix(shape={self.shape}, nnz={len(self.data)})"
