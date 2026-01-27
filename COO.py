from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.values = list(data)
        self.row_indices = list(row)
        self.col_indices = list(col)

    def to_dense(self) -> DenseMatrix:
        """Превращаем разреженную матрицу в обычный двумерный список."""
        n_rows, n_cols = self.shape
        # Создаем сетку из нулей
        grid = [[0.0 for _ in range(n_cols)] for _ in range(n_rows)]

        # Заполняем данными, суммируя значения при совпадении индексов
        for v, r, c in zip(self.values, self.row_indices, self.col_indices):
            grid[r][c] += v
        return grid

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение: просто объединяем списки координат."""
        # Если вторая матрица не COO, переводим её (через плотный вид, если нет спец. метода)
        if not isinstance(other, COOMatrix):
            other_coo = COOMatrix.from_dense(other.to_dense())
        else:
            other_coo = other

        merged_data = self.values + other_coo.values
        merged_rows = self.row_indices + other_coo.row_indices
        merged_cols = self.col_indices + other_coo.col_indices

        return COOMatrix(merged_data, merged_rows, merged_cols, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение всех ненулевых элементов на число."""
        scaled_vals = [val * scalar for val in self.values]
        return COOMatrix(scaled_vals, self.row_indices, self.col_indices, self.shape)

    def transpose(self) -> 'Matrix':
        """Меняем местами индексы строк и столбцов."""
        new_shape = (self.shape[1], self.shape[0])
        return COOMatrix(self.values, self.col_indices, self.row_indices, new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение матриц через промежуточный плотный формат (для простоты COO)."""
        # Эффективнее делать через CSR, но здесь реализуем базовый вариант
        a_dense = self.to_dense()
        b_dense = other.to_dense()

        rows_a, cols_a = self.shape
        cols_b = other.shape[1]

        res = [[0.0 for _ in range(cols_b)] for _ in range(rows_a)]
        for i in range(rows_a):
            for k in range(cols_a):
                if a_dense[i][k] != 0:
                    for j in range(cols_b):
                        res[i][j] += a_dense[i][k] * b_dense[k][j]

        return COOMatrix.from_dense(res)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создаем COO, проходя по всем ячейкам плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0

        d, r, c = [], [], []
        for i in range(rows):
            for j in range(cols):
                val = dense_matrix[i][j]
                if val != 0:
                    d.append(val)
                    r.append(i)
                    c.append(j)
        return cls(d, r, c, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """Конвертация в CSC: сортируем по столбцам."""
        from CSC import CSCMatrix
        # Сначала суммируем дубликаты
        coords = {}
        for r, c, v in zip(self.row_indices, self.col_indices, self.values):
            coords[(r, c)] = coords.get((r, c), 0.0) + v

        # Сортируем: столбец -> строка
        sorted_items = sorted(coords.items(), key=lambda x: (x[0][1], x[0][0]))

        csc_data = [v for k, v in sorted_items]
        csc_indices = [k[0] for k, v in sorted_items]

        n_cols = self.shape[1]
        ptr = [0] * (n_cols + 1)
        for k, v in sorted_items:
            ptr[k[1] + 1] += 1
        for i in range(n_cols):
            ptr[i + 1] += ptr[i]

        return CSCMatrix(csc_data, csc_indices, ptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """Конвертация в CSR: сортируем по строкам."""
        from CSR import CSRMatrix
        coords = {}
        for r, c, v in zip(self.row_indices, self.col_indices, self.values):
            coords[(r, c)] = coords.get((r, c), 0.0) + v

        # Сортируем: строка -> столбец
        sorted_items = sorted(coords.items(), key=lambda x: (x[0][0], x[0][1]))

        csr_data = [v for k, v in sorted_items]
        csr_indices = [k[1] for k, v in sorted_items]

        n_rows = self.shape[0]
        ptr = [0] * (n_rows + 1)
        for k, v in sorted_items:
            ptr[k[0] + 1] += 1
        for i in range(n_rows):
            ptr[i + 1] += ptr[i]

        return CSRMatrix(csr_data, csr_indices, ptr, self.shape)