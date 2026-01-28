# file: COO.py
from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix
from typing import Dict, Tuple, List

# используется для корректной проверки типов
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from CSC import CSCMatrix
    from CSR import CSRMatrix


class COOMatrix(Matrix):
    def __init__(self, values: COOData, row_indices: COORows, column_indices: COOCols, dims: Shape):
        super().__init__(dims)

        # проверка согласованности входных данных
        if len(values) != len(row_indices) or len(values) != len(column_indices):
            raise ValueError("Количество элементов в values, row_indices, column_indices должно совпадать")
        
        self.values = values
        self.row_idx = row_indices
        self.col_idx = column_indices
        self.dims = dims

    def to_dense(self) -> DenseMatrix:
        """Конвертирует разреженную COO‑матрицу в плотный формат."""
        n_rows, n_cols = self.dims
        result = [[0] * n_cols for _ in range(n_rows)]

        for val, r, c in zip(self.values, self.row_idx, self.col_idx):
            result[r][c] = val
        
        return result

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Реализация сложения двух COO‑матриц."""
        # используем словарь для накопления сумм
        accumulator: Dict[Tuple[int, int], float] = {}
        
        # добавляем элементы первой матрицы
        for v, r, c in zip(self.values, self.row_idx, self.col_idx):
            accumulator[(r, c)] = accumulator.get((r, c), 0.0) + v
        
        # добавляем элементы второй матрицы
        for v, r, c in zip(other.values, other.row_idx, other.col_idx):
            accumulator[(r, c)] = accumulator.get((r, c), 0.0) + v
        
        # собираем только ненулевые элементы
        new_vals, new_rows, new_cols = [], [], []
        for (r, c), total in accumulator.items():
            if abs(total) > 1e-14:
                new_vals.append(total)
                new_rows.append(r)
                new_cols.append(c)
        
        return COOMatrix(new_vals, new_rows, new_cols, self.dims)

    def _mul_impl(self, factor: float) -> 'Matrix':
        """Умножение матрицы на число."""
        scaled_values = [v * factor for v in self.values]
        return COOMatrix(scaled_values, self.row_idx[:], self.col_idx[:], self.dims)

    def transpose(self) -> 'Matrix':
        """Возвращает транспонированную матрицу."""
        new_dims = (self.dims[1], self.dims[0])
        return COOMatrix(self.values.copy(), self.col_idx.copy(), self.row_idx.copy(), new_dims)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Реализация умножения матриц."""
        if self.dims[1] != other.dims[0]:
            raise ValueError("Несовместимые размеры матриц для умножения")
        
        m, n = self.dims[0], other.dims[1]
        temp_result = {}

        # конвертируем правую матрицу в CSR для эффективного доступа
        other_csr = other._to_csr()

        for idx in range(len(self.values)):
            r = self.row_idx[idx]
            c = self.col_idx[idx]
            v_a = self.values[idx]

            start = other_csr.indptr[c]
            stop = other_csr.indptr[c + 1]

            for pos in range(start, stop):
                col_b = other_csr.indices[pos]
                v_b = other_csr.data[pos]
                key = (r, col_b)
                temp_result[key] = temp_result.get(key, 0.0) + v_a * v_b

        # преобразуем результат обратно в COO
        res_vals, res_rows, res_cols = [], [], []
        for (r, c), val in temp_result.items():
            if abs(val) > 1e-14:
                res_vals.append(val)
                res_rows.append(r)
                res_cols.append(c)
        
        return COOMatrix(res_vals, res_rows, res_cols, (m, n))

    @classmethod
    def from_dense(cls, dense: DenseMatrix) -> 'COOMatrix':
        """Создаёт COO‑матрицу из плотного представления."""
        vals, rows, cols = [], [], []
        n_rows = len(dense)
        n_cols = len(dense[0]) if n_rows > 0 else 0

        for r in range(n_rows):
            for c in range(n_cols):
                elem = dense[r][c]
                if elem != 0:
                    vals.append(elem)
                    rows.append(r)
                    cols.append(c)
        
        return cls(vals, rows, cols, (n_rows, n_cols))

    def _to_csc(self) -> 'CSCMatrix':
        """Конвертирует текущую матрицу в формат CSC."""
        from CSC import CSCMatrix
        n_rows, n_cols = self.dims
        
        # сортируем элементы по столбцам, затем по строкам
        entries = list(zip(self.col_idx, self.row_idx, self.values))
        entries.sort()
        
        csc_vals = []
        csc_rows = []
        csc_colptr = [0] * (n_cols + 1)
        
        for col, row, val in entries:
            csc_vals.append(val)
            csc_rows.append(row)
            csc_colptr[col + 1] += 1
        
        # аккумулируем указатели столбцов
        for j in range(n_cols):
            csc_colptr[j + 1] += csc_colptr[j]
        
        return CSCMatrix(csc_vals, csc_rows, csc_colptr, self.dims)

    def _to_csr(self) -> 'CSRMatrix':
        """Конвертирует текущую матрицу в формат CSR."""
        from CSR import CSRMatrix
        n_rows, n_cols = self.dims
        
        # сортируем элементы по строкам, затем по столбцам
        entries = list(zip(self.row_idx, self.col_idx, self.values))
        entries.sort()
        
        csr_vals = []
        csr_cols = []
        csr_rowptr = [0] * (n_rows + 1)
        
        for row, col, val in entries:
            csr_vals.append(val)
            csr_cols.append(col)
            csr_rowptr[row + 1] += 1
        
        # аккумулируем указатели строк
        for i in range(n_rows):
            csr_rowptr[i + 1] += csr_rowptr[i]
        
        return CSRMatrix(csr_vals, csr_cols, csr_rowptr, self.dims)