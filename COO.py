from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix
from typing import Dict, List, Tuple
import bisect


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.row = row
        self.col = col
        if not (len(data) == len(row) == len(col)):
            raise ValueError("Длины data, row и col должны совпадать")

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]
        
        for value, i, j in zip(self.data, self.row, self.col):
            if 0 <= i < rows and 0 <= j < cols:
                dense[i][j] = value
        
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц."""
        if not isinstance(other, COOMatrix):
            # Конвертируем other в COO
            other_coo = COOMatrix.from_dense(other.to_dense())
            return self._add_impl(other_coo)
        
        # Эффективное сложение двух COO матриц
        # Создаем словарь для быстрого доступа: (row, col) -> индекс
        index_map = {}
        for idx, (r, c) in enumerate(zip(self.row, self.col)):
            index_map[(r, c)] = idx
        
        new_data = self.data.copy()
        new_row = self.row.copy()
        new_col = self.col.copy()
        
        for val, r, c in zip(other.data, other.row, other.col):
            key = (r, c)
            if key in index_map:
                idx = index_map[key]
                new_data[idx] += val
            else:
                new_data.append(val)
                new_row.append(r)
                new_col.append(c)
                index_map[key] = len(new_data) - 1
        
        return COOMatrix(new_data, new_row, new_col, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        new_data = [value * scalar for value in self.data]
        return COOMatrix(new_data, self.row.copy(), self.col.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        return COOMatrix(
            self.data.copy(),
            self.col.copy(),
            self.row.copy(),
            (self.shape[1], self.shape[0])
        )

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""
        # Эффективное умножение разреженных матриц
        if isinstance(other, COOMatrix):
            return self._matmul_coo(other)
        else:
            # Конвертируем в COO и умножаем
            other_coo = COOMatrix.from_dense(other.to_dense())
            return self._matmul_coo(other_coo)

    def _matmul_coo(self, other: 'COOMatrix') -> 'COOMatrix':
        """Умножение двух COO матриц."""
        rows_A, cols_A = self.shape
        rows_B, cols_B = other.shape
        
        if cols_A != rows_B:
            raise ValueError("Несовместимые размерности для умножения")
        
        # Создаем структуру для хранения результата построчно
        # row_dict[i] = {col: value}
        row_dicts = [{} for _ in range(rows_A)]
        
        # Для эффективности создаем представление other по строкам
        other_by_row = {}
        for idx, (r, c) in enumerate(zip(other.row, other.col)):
            if r not in other_by_row:
                other_by_row[r] = []
            other_by_row[r].append((c, other.data[idx]))
        
        # Умножаем
        for idx, (i, k) in enumerate(zip(self.row, self.col)):
            val_A = self.data[idx]
            if k in other_by_row:
                for j, val_B in other_by_row[k]:
                    if j in row_dicts[i]:
                        row_dicts[i][j] += val_A * val_B
                    else:
                        row_dicts[i][j] = val_A * val_B
        
        # Преобразуем row_dicts в COO формат
        data = []
        row = []
        col = []
        
        for i in range(rows_A):
            for j, val in row_dicts[i].items():
                if abs(val) > 1e-12:  # Игнорируем очень маленькие значения
                    data.append(val)
                    row.append(i)
                    col.append(j)
        
        return COOMatrix(data, row, col, (rows_A, cols_B))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        data = []
        row = []
        col = []
        
        for i, row_vals in enumerate(dense_matrix):
            for j, val in enumerate(row_vals):
                if abs(val) > 1e-12:  # Сохраняем только значимые значения
                    data.append(val)
                    row.append(i)
                    col.append(j)
        
        shape = (len(dense_matrix), len(dense_matrix[0]) if dense_matrix else 0)
        return cls(data, row, col, shape)

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        # Импортируем тут, а не в начале файла
        from CSC import CSCMatrix
        
        if not self.data:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)
        
        n_cols = self.shape[1]
        
        # Сортируем по столбцам, затем по строкам
        sorted_indices = sorted(range(len(self.data)), 
                               key=lambda i: (self.col[i], self.row[i]))
        
        data = [self.data[i] for i in sorted_indices]
        indices = [self.row[i] for i in sorted_indices]
        
        # Строим indptr
        indptr = [0] * (n_cols + 1)
        
        for col_idx in self.col:
            indptr[col_idx + 1] += 1
        
        # Накопительная сумма
        for i in range(1, n_cols + 1):
            indptr[i] += indptr[i - 1]
        
        return CSCMatrix(data, indices, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        # Импортируем тут, а не в начале файла
        from CSR import CSRMatrix
        
        if not self.data:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)
        
        n_rows = self.shape[0]
        
        # Сортируем по строкам, затем по столбцам
        sorted_indices = sorted(range(len(self.data)), 
                               key=lambda i: (self.row[i], self.col[i]))
        
        data = [self.data[i] for i in sorted_indices]
        indices = [self.col[i] for i in sorted_indices]
        
        # Строим indptr
        indptr = [0] * (n_rows + 1)
        
        for row_idx in self.row:
            indptr[row_idx + 1] += 1
        
        # Накопительная сумма
        for i in range(1, n_rows + 1):
            indptr[i] += indptr[i - 1]
        
        return CSRMatrix(data, indices, indptr, self.shape)
    
    def get_element(self, i: int, j: int) -> float:
        """Получить элемент (i, j)."""
        for idx, (r, c) in enumerate(zip(self.row, self.col)):
            if r == i and c == j:
                return self.data[idx]
        return 0.0