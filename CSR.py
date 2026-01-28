# file: CSR.py
from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix

# используется для корректной проверки типов
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from CSC import CSCMatrix
    from COO import COOMatrix


class CSRMatrix(Matrix):
    def __init__(self, values: CSRData, col_idxs: CSRIndices, row_ptrs: CSRIndptr, dims: Shape):
        super().__init__(dims)

        # проверки корректности структуры CSR
        if len(row_ptrs) != dims[0] + 1:
            raise ValueError(f"Длина row_ptrs должна быть {dims[0] + 1}")
        if row_ptrs[0] != 0:
            raise ValueError("Первый элемент row_ptrs должен быть равен 0")
        if row_ptrs[-1] != len(values):
            raise ValueError(f"Последний элемент row_ptrs должен быть равен {len(values)}")
        if len(values) != len(col_idxs):
            raise ValueError("Длины values и col_idxs должны совпадать")
        
        self.vals = values
        self.cols = col_idxs
        self.row_ptr = row_ptrs

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR‑матрицу в плотный формат."""
        n_rows, n_cols = self.dims
        dense = [[0] * n_cols for _ in range(n_rows)]
        
        for i in range(n_rows):
            start = self.row_ptr[i]
            end = self.row_ptr[i + 1]
            for idx in range(start, end):
                j = self.cols[idx]
                val = self.vals[idx]
                dense[i][j] = val
        
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение двух CSR‑матриц через промежуточное COO‑представление."""
        coo_self = self._to_coo()
        coo_other = other._to_coo()
        coo_sum = coo_self._add_impl(coo_other)
        return coo_sum._to_csr()

    def _mul_impl(self, factor: float) -> 'Matrix':
        """Умножение матрицы на скаляр."""
        if abs(factor) < 1e-14:
            # результат – нулевая матрица той же размерности
            return CSRMatrix([], [], [0] * (self.dims[0] + 1), self.dims)
        
        scaled_vals = [v * factor for v in self.vals]
        return CSRMatrix(scaled_vals, self.cols[:], self.row_ptr[:], self.dims)

    def transpose(self) -> 'Matrix':
        """Транспонирование матрицы. Результат возвращается в формате CSC."""
        from CSC import CSCMatrix
        old_rows, old_cols = self.dims
        new_rows, new_cols = old_cols, old_rows
        
        # подсчёт количества элементов в каждом столбце результата
        col_sizes = [0] * new_cols
        for i in range(old_rows):
            col_sizes[i] = self.row_ptr[i + 1] - self.row_ptr[i]
        
        new_colptr = [0] * (new_cols + 1)
        for j in range(new_cols):
            new_colptr[j + 1] = new_colptr[j] + col_sizes[j]
        
        result_vals = [0] * len(self.vals)
        result_rows = [0] * len(self.cols)
        fill_positions = new_colptr.copy()
        
        for i in range(old_rows):
            start = self.row_ptr[i]
            end = self.row_ptr[i + 1]
            for idx in range(start, end):
                j = self.cols[idx]
                val = self.vals[idx]
                
                pos = fill_positions[i]
                result_vals[pos] = val
                result_rows[pos] = j
                fill_positions[i] += 1
        
        return CSCMatrix(result_vals, result_rows, new_colptr, (new_rows, new_cols))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR‑матрицы на другую матрицу."""
        m, n = self.dims[0], other.dims[1]
        
        result_vals = []
        result_cols = []
        result_rowptr = [0] * (m + 1)
        
        for i in range(m):
            row_accumulator = {}
            
            a_start = self.row_ptr[i]
            a_end = self.row_ptr[i + 1]
            
            for a_idx in range(a_start, a_end):
                k = self.cols[a_idx]
                a_val = self.vals[a_idx]
                
                b_start = other.row_ptr[k]
                b_end = other.row_ptr[k + 1]
                
                for b_idx in range(b_start, b_end):
                    j = other.cols[b_idx]
                    b_val = other.vals[b_idx]
                    
                    row_accumulator[j] = row_accumulator.get(j, 0.0) + a_val * b_val
            
            # сохраняем ненулевые элементы строки результата
            sorted_columns = sorted(row_accumulator.keys())
            for j in sorted_columns:
                val = row_accumulator[j]
                if abs(val) > 1e-14:
                    result_vals.append(val)
                    result_cols.append(j)
            
            result_rowptr[i + 1] = len(result_vals)
        
        return CSRMatrix(result_vals, result_cols, result_rowptr, (m, n))

    @classmethod
    def from_dense(cls, dense: DenseMatrix) -> 'CSRMatrix':
        """Создаёт CSR‑матрицу из плотного представления."""
        n_rows = len(dense)
        n_cols = len(dense[0]) if n_rows > 0 else 0
        
        vals = []
        cols = []
        row_counts = [0] * n_rows
        
        for r in range(n_rows):
            for c in range(n_cols):
                elem = dense[r][c]
                if elem != 0:
                    vals.append(elem)
                    cols.append(c)
                    row_counts[r] += 1
        
        row_ptr = [0] * (n_rows + 1)
        for i in range(n_rows):
            row_ptr[i + 1] = row_ptr[i] + row_counts[i]
        
        return cls(vals, cols, row_ptr, (n_rows, n_cols))

    def _to_csc(self) -> 'CSCMatrix':
        """Конвертирует CSR‑матрицу в формат CSC."""
        from CSC import CSCMatrix
        m, n = self.dims
        
        # подсчёт количества элементов в каждом столбце
        col_counts = [0] * n
        for c in self.cols:
            col_counts[c] += 1
        
        col_ptr = [0] * (n + 1)
        for j in range(n):
            col_ptr[j + 1] = col_ptr[j] + col_counts[j]
        
        vals = [0] * len(self.vals)
        rows = [0] * len(self.cols)
        insert_pos = col_ptr.copy()
        
        for i in range(m):
            start = self.row_ptr[i]
            end = self.row_ptr[i + 1]
            for idx in range(start, end):
                j = self.cols[idx]
                val = self.vals[idx]
                
                pos = insert_pos[j]
                vals[pos] = val
                rows[pos] = i
                insert_pos[j] += 1
        
        return CSCMatrix(vals, rows, col_ptr, (m, n))

    def _to_coo(self) -> 'COOMatrix':
        """Конвертирует CSR‑матрицу в формат COO."""
        from COO import COOMatrix
        n_rows, n_cols = self.dims
        
        coo_vals = []
        coo_rows = []
        coo_cols = []
        
        for i in range(n_rows):
            start = self.row_ptr[i]
            end = self.row_ptr[i + 1]
            for idx in range(start, end):
                j = self.cols[idx]
                val = self.vals[idx]
                coo_vals.append(val)
                coo_rows.append(i)
                coo_cols.append(j)
        
        return COOMatrix(coo_vals, coo_rows, coo_cols, self.dims)