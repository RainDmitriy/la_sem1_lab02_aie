from base import Matrix
from types import COOData, COORows, COOCols, Shape, DenseMatrix
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from CSR import CSRMatrix
    from CSC import CSCMatrix

# Порог для удаления численного шума после операций
ZERO_TOL = 1e-12


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        """
        Создаёт COO-матрицу. Хранит ВСЕ переданные тройки (value, row, col),
        даже если value == 0.0 или индексы дублируются.
        """
        super().__init__(shape)

        if not (len(data) == len(row) == len(col)):
            raise ValueError("Длина data, row и col должна совпадать")

        rows, cols = shape

        # Проверка корректности индексов
        for r, c in zip(row, col):
            if rows == 0 or cols == 0:
                raise ValueError(f"Матрица размера {shape} не может содержать элементы")
            if not (0 <= r < rows and 0 <= c < cols):
                raise ValueError(f"Индекс ({r}, {c}) выходит за пределы матрицы {shape}")

        self.data = list(data)
        self.row = list(row)
        self.col = list(col)

    def eliminate_zeros(self, tol: float = ZERO_TOL) -> 'COOMatrix':
        """
        Возвращает новую COO-матрицу без элементов, близких к нулю.
        Используется после арифметических операций для подавления численного шума.
        """
        mask = [abs(v) > tol for v in self.data]
        new_data = [v for v, m in zip(self.data, mask) if m]
        new_row  = [r for r, m in zip(self.row,  mask) if m]
        new_col  = [c for c, m in zip(self.col,  mask) if m]
        return COOMatrix(new_data, new_row, new_col, self.shape)

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу. Дубликаты индексов суммируются."""
        rows, cols = self.shape
        dense = [[0.0 for _ in range(cols)] for _ in range(rows)]

        for value, r, c in zip(self.data, self.row, self.col):
            dense[r][c] += value

        return dense

    def _add_impl(self, other: 'Matrix') -> 'COOMatrix':
        """Сложение. Результат очищается от численного шума."""
        if self.shape != other.shape:
            raise ValueError(f"Несовместимые размеры: {self.shape} и {other.shape}")

        result = {}

        for v, r, c in zip(self.data, self.row, self.col):
            result[(r, c)] = result.get((r, c), 0.0) + v

        if isinstance(other, COOMatrix):
            other_triplets = zip(other.data, other.row, other.col)
        else:
            dense = other.to_dense()
            other_triplets = (
                (dense[i][j], i, j)
                for i in range(self.shape[0])
                for j in range(self.shape[1])
                if dense[i][j] != 0.0
            )

        for v, r, c in other_triplets:
            result[(r, c)] = result.get((r, c), 0.0) + v

        if not result:
            return COOMatrix([], [], [], self.shape)

        items = sorted(result.items())
        data = [val for (_, _), val in items]
        row  = [r   for (r, _), _   in items]
        col  = [c   for (_, c), _   in items]

        return COOMatrix(data, row, col, self.shape).eliminate_zeros()

    def _mul_impl(self, scalar: float) -> 'COOMatrix':
        """Умножение на скаляр. Результат очищается от шума."""
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Скаляр должен быть числом, получен {type(scalar)}")

        new_data = [v * scalar for v in self.data]
        result = COOMatrix(new_data, self.row, self.col, self.shape)
        return result.eliminate_zeros() if scalar != 0 else result

    def transpose(self) -> 'COOMatrix':
        """Транспонирование."""
        return COOMatrix(self.data, self.col, self.row, (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'COOMatrix':
        """Умножение матриц. Поддерживает только COO × COO напрямую."""
        if not isinstance(other, COOMatrix):
            other = COOMatrix.from_dense(other.to_dense())

        if self.shape[1] != other.shape[0]:
            raise ValueError(f"Несовместимые размеры: {self.shape} и {other.shape}")

        right_by_row = {}
        for v, r, c in zip(other.data, other.row, other.col):
            right_by_row.setdefault(r, []).append((c, v))

        result = {}
        for a_val, a_row, a_col in zip(self.data, self.row, self.col):
            if a_col in right_by_row:
                for b_col, b_val in right_by_row[a_col]:
                    key = (a_row, b_col)
                    result[key] = result.get(key, 0.0) + a_val * b_val

        if not result:
            return COOMatrix([], [], [], (self.shape[0], other.shape[1]))

        items = sorted(result.items())
        data = [val for (_, _), val in items]
        row  = [r   for (r, _), _   in items]
        col  = [c   for (_, c), _   in items]

        return COOMatrix(data, row, col, (self.shape[0], other.shape[1])).eliminate_zeros()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создаёт COO из плотной матрицы."""
        if not dense_matrix:
            return cls([], [], [], (0, 0))

        rows = len(dense_matrix)
        cols = len(dense_matrix[0])

        data, row_idx, col_idx = [], [], []
        for i in range(rows):
            for j in range(cols):
                if dense_matrix[i][j] != 0.0:
                    data.append(dense_matrix[i][j])
                    row_idx.append(i)
                    col_idx.append(j)

        return cls(data, row_idx, col_idx, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """Преобразует в CSC. Суммирует дубликаты, сортирует строки внутри столбцов."""
        from CSC import CSCMatrix

        rows, cols = self.shape
        col_dict = [{} for _ in range(cols)]
        for v, r, c in zip(self.data, self.row, self.col):
            col_dict[c][r] = col_dict[c].get(r, 0.0) + v

        data, indices, indptr = [], [], [0]
        for c in range(cols):
            if col_dict[c]:
                for r in sorted(col_dict[c].keys()):
                    val = col_dict[c][r]
                    if abs(val) > ZERO_TOL:
                        data.append(val)
                        indices.append(r)
            indptr.append(len(data))

        return CSCMatrix(data, indices, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """Преобразует в CSR. Суммирует дубликаты, сортирует столбцы внутри строк."""
        from CSR import CSRMatrix

        rows, cols = self.shape
        row_dict = [{} for _ in range(rows)]
        for v, r, c in zip(self.data, self.row, self.col):
            row_dict[r][c] = row_dict[r].get(c, 0.0) + v

        data, indices, indptr = [], [], [0]
        for r in range(rows):
            if row_dict[r]:
                for c in sorted(row_dict[r].keys()):
                    val = row_dict[r][c]
                    if abs(val) > ZERO_TOL:
                        data.append(val)
                        indices.append(c)
            indptr.append(len(data))

        return CSRMatrix(data, indices, indptr, self.shape)