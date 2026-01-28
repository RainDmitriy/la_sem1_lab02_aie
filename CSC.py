# file: CSC.py
from base import Matrix
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix

# используется для корректной проверки типов
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from COO import COOMatrix
    from CSR import CSRMatrix


class CSCMatrix(Matrix):
    def __init__(self, values: CSCData, row_idxs: CSCIndices, col_ptrs: CSCIndptr, dims: Shape):
        super().__init__(dims)

        # проверки корректности структуры CSC
        if len(col_ptrs) != dims[1] + 1:
            raise ValueError(f"Длина col_ptrs должна быть {dims[1] + 1}")
        if col_ptrs[0] != 0:
            raise ValueError("Первый элемент col_ptrs должен быть равен 0")
        if col_ptrs[-1] != len(values):
            raise ValueError(f"Последний элемент col_ptrs должен быть равен {len(values)}")
        if len(values) != len(row_idxs):
            raise ValueError("Длины values и row_idxs должны совпадать")
        
        self.vals = values
        self.rows = row_idxs
        self.col_ptr = col_ptrs

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC‑матрицу в плотный формат."""
        n_rows, n_cols = self.dims
        dense = [[0] * n_cols for _ in range(n_rows)]
        
        for col in range(n_cols):
            start = self.col_ptr[col]
            end = self.col_ptr[col + 1]
            for idx in range(start, end):
                row = self.rows[idx]
                val = self.vals[idx]
                dense[row][col] = val
        
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение двух CSC‑матриц через промежуточное COO‑представление."""
        coo_self = self._to_coo()
        coo_other = other._to_coo()
        coo_sum = coo_self._add_impl(coo_other)
        return coo_sum._to_csc()

    def _mul_impl(self, factor: float) -> 'Matrix':
        """Умножение матрицы на скаляр."""
        if abs(factor) < 1e-14:
            # результат – нулевая матрица той же размерности
            return CSCMatrix([], [], [0] * (self.dims[1] + 1), self.dims)
        
        scaled_vals = [v * factor for v in self.vals]
        return CSCMatrix(scaled_vals, self.rows[:], self.col_ptr[:], self.dims)

    def transpose(self) -> 'Matrix':
        """Транспонирование матрицы. Результат возвращается в формате CSR."""
        from CSR import CSRMatrix
        old_rows, old_cols = self.dims
        new_rows, new_cols = old_cols, old_rows
        
        # подсчёт количества элементов в каждой строке результата
        row_sizes = [0] * new_rows
        for col in range(old_cols):
            row_sizes[col] = self.col_ptr[col + 1] - self.col_ptr[col]
        
        new_rowptr = [0] * (new_rows + 1)
        for i in range(new_rows):
            new_rowptr[i + 1] = new_rowptr[i] + row_sizes[i]
        
        result_vals = [0] * len(self.vals)
        result_cols = [0] * len(self.rows)
        fill_positions = new_rowptr.copy()
        
        for col in range(old_cols):
            start = self.col_ptr[col]
            end = self.col_ptr[col + 1]
            for idx in range(start, end):
                row = self.rows[idx]
                val = self.vals[idx]
                
                pos = fill_positions[col]
                result_vals[pos] = val
                result_cols[pos] = row
                fill_positions[col] += 1
        
        return CSRMatrix(result_vals, result_cols, new_rowptr, (new_rows, new_cols))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC‑матрицы на другую матрицу."""
        m, n = self.dims[0], other.dims[1]
        
        result_vals = []
        result_rows = []
        result_colptr = [0] * (n + 1)
        
        # предподготовка данных второй матрицы по строкам
        rows_of_B = [[] for _ in range(other.dims[0])]
        for col in range(n):
            start = other.col_ptr[col]
            end = other.col_ptr[col + 1]
            for idx in range(start, end):
                row = other.rows[idx]
                val = other.vals[idx]
                rows_of_B[row].append((col, val))
        
        # временный массив для накопления строки результата
        temp_row = [0.0] * m
        
        for j in range(n):
            # обнуляем временный массив
            for i in range(m):
                temp_row[i] = 0.0
            
            # вычисляем j‑й столбец результата
            for i in range(other.dims[0]):
                for col_b, val_b in rows_of_B[i]:
                    if col_b == j:
                        col_start = self.col_ptr[i]
                        col_end = self.col_ptr[i + 1]
                        for a_idx in range(col_start, col_end):
                            row_a = self.rows[a_idx]
                            val_a = self.vals[a_idx]
                            temp_row[row_a] += val_a * val_b
            
            # сохраняем ненулевые элементы столбца
            for i in range(m):
                if abs(temp_row[i]) > 1e-14:
                    result_vals.append(temp_row[i])
                    result_rows.append(i)
            
            result_colptr[j + 1] = len(result_vals)
        
        return CSCMatrix(result_vals, result_rows, result_colptr, (m, n))

    @classmethod
    def from_dense(cls, dense: DenseMatrix) -> 'CSCMatrix':
        """Создаёт CSC‑матрицу из плотного представления."""
        n_rows = len(dense)
        n_cols = len(dense[0]) if n_rows > 0 else 0
        
        vals = []
        rows = []
        col_counts = [0] * n_cols
        
        for c in range(n_cols):
            for r in range(n_rows):
                elem = dense[r][c]
                if elem != 0:
                    vals.append(elem)
                    rows.append(r)
                    col_counts[c] += 1
        
        col_ptr = [0] * (n_cols + 1)
        for j in range(n_cols):
            col_ptr[j + 1] = col_ptr[j] + col_counts[j]
        
        return cls(vals, rows, col_ptr, (n_rows, n_cols))

    def _to_csr(self) -> 'CSRMatrix':
        """Конвертирует CSC‑матрицу в формат CSR."""
        from CSR import CSRMatrix
        m, n = self.dims
        
        # подсчёт количества элементов в каждой строке
        row_counts = [0] * m
        for r in self.rows:
            row_counts[r] += 1
        
        row_ptr = [0] * (m + 1)
        for i in range(m):
            row_ptr[i + 1] = row_ptr[i] + row_counts[i]
        
        vals = [0] * len(self.vals)
        cols = [0] * len(self.rows)
        insert_pos = row_ptr.copy()
        
        for j in range(n):
            start = self.col_ptr[j]
            end = self.col_ptr[j + 1]
            for idx in range(start, end):
                i = self.rows[idx]
                val = self.vals[idx]
                
                pos = insert_pos[i]
                vals[pos] = val
                cols[pos] = j
                insert_pos[i] += 1
        
        return CSRMatrix(vals, cols, row_ptr, (m, n))

    def _to_coo(self) -> 'COOMatrix':
        """Конвертирует CSC‑матрицу в формат COO."""
        from COO import COOMatrix
        n_rows, n_cols = self.dims
        
        coo_vals = []
        coo_rows = []
        coo_cols = []
        
        for col in range(n_cols):
            start = self.col_ptr[col]
            end = self.col_ptr[col + 1]
            for idx in range(start, end):
                row = self.rows[idx]
                val = self.vals[idx]
                coo_vals.append(val)
                coo_rows.append(row)
                coo_cols.append(col)
        
        return COOMatrix(coo_vals, coo_rows, coo_cols, self.dims)