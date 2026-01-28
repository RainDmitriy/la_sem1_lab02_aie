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

        coo_self = self._to_coo()
        coo_other = other._to_coo()

        res = coo_self._add_impl(coo_other)

        return res._to_csc()

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
        assert isinstance(other, CSCMatrix), "Multiplication is only supported between two CSC matrices"

        n_rows, n_cols = self.shape
        k = other.shape[1]

        self_csr = self._to_csr()

        result_data = [[0.0 for _ in range(k)] for _ in range(n_rows)]

        for i in range(n_rows):
            row_start = self_csr.indptr[i]
            row_end = self_csr.indptr[i + 1]

            if row_start == row_end:
                continue

            for j in range(k):
                col_start = other.indptr[j]
                col_end = other.indptr[j + 1]

                if col_start == col_end:
                    continue

                dot = 0.0

                p1, p2 = row_start, col_start
                while p1 < row_end and p2 < col_end:
                    col_in_self = self_csr.indices[p1]
                    row_in_other = other.indices[p2]

                    if col_in_self < row_in_other:
                        p1 += 1
                    elif col_in_self > row_in_other:
                        p2 += 1
                    else:
                        # Нашли совпадающий индекс k
                        dot += self_csr.data[p1] * other.data[p2]
                        p1 += 1
                        p2 += 1

                if fabs(dot) >= self.ZERO_TOLERANCE:
                    result_data[i][j] = dot

        return CSCMatrix.from_dense(result_data)

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
        new_n_rows, new_n_cols = n_cols, n_rows

        row_counts = [0] * new_n_rows

        for row_idx in self.indices:
            row_counts[row_idx] += 1

        row_ptr = [0] * (n_rows + 1)
        for i in range(new_n_rows):
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
        return CSRMatrix(new_data, new_indices, row_ptr, (new_n_rows, new_n_cols))

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
