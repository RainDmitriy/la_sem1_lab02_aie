# COO.py
from base import Matrix
from my_types import COOData, COORows, COOCols, Shape, DenseMatrix


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.row = row
        self.col = col

    def to_dense(self) -> DenseMatrix:
        rows, cols = self.shape
        res = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for i in range(len(self.data)):
            res[self.row[i]][self.col[i]] = self.data[i]
        return res

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        # Вместо плотных матриц просто объединяем данные.
        # Формат COO допускает несколько записей для одной ячейки.
        if not hasattr(other, 'row'):  # Если это не COO, приводим к нему
            other = other._to_coo()

        return COOMatrix(
            self.data + other.data,
            self.row + other.row,
            self.col + other.col,
            self.shape
        )

    def _mul_impl(self, scalar: float) -> 'Matrix':
        new_data = [val * scalar for val in self.data]
        return COOMatrix(new_data, self.row, self.col, self.shape)

    def transpose(self) -> 'Matrix':
        return COOMatrix(self.data, self.col, self.row, (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        # Быстрое умножение через индексацию строк первой матрицы
        if not hasattr(other, 'row'):
            other = other._to_coo()

        # Индексируем первую матрицу: {row: {col: value}}
        left_map = {}
        for i in range(len(self.data)):
            r, c, v = self.row[i], self.col[i], self.data[i]
            if r not in left_map:
                left_map[r] = {}
            left_map[r][c] = left_map[r].get(c, 0.0) + v

        res_data, res_row, res_col = [], [], []

        # Чтобы не суммировать дубликаты в процессе, используем временный накопитель
        # для текущей строки: {col: sum_value}
        for r1, row_items in left_map.items():
            temp_res = {}
            for i in range(len(other.data)):
                r2, c2, v2 = other.row[i], other.col[i], other.data[i]
                if r2 in row_items:
                    val = row_items[r2] * v2
                    temp_res[c2] = temp_res.get(c2, 0.0) + val

            # Переносим накопленные данные строки в итоговые массивы
            for col_idx, final_val in temp_res.items():
                if final_val != 0:
                    res_data.append(final_val)
                    res_row.append(r1)
                    res_col.append(col_idx)

        return COOMatrix(res_data, res_row, res_col, (self.shape[0], other.shape[1]))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        data, r_idx, c_idx = [], [], []
        for r in range(rows):
            for c in range(cols):
                if dense_matrix[r][c] != 0:
                    data.append(dense_matrix[r][c])
                    r_idx.append(r)
                    c_idx.append(c)
        return cls(data, r_idx, c_idx, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        from CSC import CSCMatrix
        return CSCMatrix.from_dense(self.to_dense())

    def _to_csr(self) -> 'CSRMatrix':
        from CSR import CSRMatrix
        return CSRMatrix.from_dense(self.to_dense())