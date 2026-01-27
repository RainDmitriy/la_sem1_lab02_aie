from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix
from typing import Dict, Tuple


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.row = row
        self.col = col
        self.nnz = len(data)

    def to_dense(self) -> DenseMatrix:
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]
        for val, r, c in zip(self.data, self.row, self.col):
            dense[r][c] = val
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц."""
        # Преобразуем обе матрицы в COO
        if not isinstance(other, COOMatrix):
            other = other._to_coo()
        
        # Используем словарь для объединения
        result_dict = {}
        
        # Добавляем элементы из текущей матрицы
        for i in range(self.nnz):
            key = (self.row[i], self.col[i])
            result_dict[key] = self.data[i]
        
        # Добавляем или суммируем элементы из другой матрицы
        for i in range(other.nnz):
            key = (other.row[i], other.col[i])
            result_dict[key] = result_dict.get(key, 0.0) + other.data[i]
        
        # Формируем отсортированные массивы
        sorted_keys = sorted(result_dict.keys())
        result_data = []
        result_row = []
        result_col = []
        
        for r, c in sorted_keys:
            val = result_dict[(r, c)]
            if abs(val) > 1e-7:
                result_data.append(val)
                result_row.append(r)
                result_col.append(c)
        
        return COOMatrix(result_data, result_row, result_col, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        if abs(scalar) < 1e-7:
            return COOMatrix([], [], [], self.shape)
        new_data = [val * scalar for val in self.data]
        return COOMatrix(new_data, self.row[:], self.col[:], self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        # Меняем местами строки и столбцы
        return COOMatrix(self.data[:], self.col[:], self.row[:], (self.cols, self.rows))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""
        # Используем алгоритм умножения через словари
        rows = self.rows
        cols = other.cols
        
        # Получаем данные другой матрицы в удобном формате
        if isinstance(other, COOMatrix):
            # Для COO создаем словарь по столбцам
            other_by_col = {}
            for i in range(other.nnz):
                r, c, val = other.row[i], other.col[i], other.data[i]
                if c not in other_by_col:
                    other_by_col[c] = {}
                other_by_col[c][r] = val
        else:
            # Преобразуем в COO
            other_coo = other._to_coo()
            return self._matmul_impl(other_coo)
        
        # Создаем словарь для результата
        result_dict = {}
        
        # Умножаем
        for i in range(self.nnz):
            r1, c1, val1 = self.row[i], self.col[i], self.data[i]
            if c1 in other_by_col:
                for r2, val2 in other_by_col[c1].items():
                    key = (r1, r2)
                    result_dict[key] = result_dict.get(key, 0.0) + val1 * val2
        
        # Формируем результат
        sorted_keys = sorted(result_dict.keys())
        result_data = []
        result_row = []
        result_col = []
        
        for r, c in sorted_keys:
            val = result_dict[(r, c)]
            if abs(val) > 1e-7:
                result_data.append(val)
                result_row.append(r)
                result_col.append(c)
        
        return COOMatrix(result_data, result_row, result_col, (rows, cols))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        data = []
        row = []
        col = []
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        
        for r in range(rows):
            for c in range(cols):
                val = dense_matrix[r][c]
                if abs(val) > 1e-7:
                    data.append(val)
                    row.append(r)
                    col.append(c)
        
        return cls(data, row, col, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix
        
        if self.nnz == 0:
            return CSCMatrix([], [], [0] * (self.cols + 1), self.shape)
        
        # Сортируем по столбцам, затем по строкам
        indices = list(range(self.nnz))
        indices.sort(key=lambda i: (self.col[i], self.row[i]))
        
        data = [self.data[i] for i in indices]
        indices_rows = [self.row[i] for i in indices]
        
        # Строим indptr
        indptr = [0] * (self.cols + 1)
        for i in range(self.nnz):
            col_idx = self.col[indices[i]]
            indptr[col_idx + 1] += 1
        
        for j in range(self.cols):
            indptr[j + 1] += indptr[j]
        
        return CSCMatrix(data, indices_rows, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix
        
        if self.nnz == 0:
            return CSRMatrix([], [], [0] * (self.rows + 1), self.shape)
        
        # Сортируем по строкам, затем по столбцам
        indices = list(range(self.nnz))
        indices.sort(key=lambda i: (self.row[i], self.col[i]))
        
        data = [self.data[i] for i in indices]
        indices_cols = [self.col[i] for i in indices]
        
        # Строим indptr
        indptr = [0] * (self.rows + 1)
        for i in range(self.nnz):
            row_idx = self.row[indices[i]]
            indptr[row_idx + 1] += 1
        
        for i in range(self.rows):
            indptr[i + 1] += indptr[i]
        
        return CSRMatrix(data, indices_cols, indptr, self.shape)
