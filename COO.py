from base import Matrix
from typing import List, Tuple

TOL = 1e-12


class COOMatrix(Matrix):
    def __init__(self, data: List[float], row: List[int], col: List[int], shape: Tuple[int, int]):
        super().__init__(shape)
        self.data = data
        self.row = row
        self.col = col
        self.nnz = len(data)
        
        # Проверка, что все индексы в пределах
        if row and max(row) >= shape[0]:
            raise ValueError(f"Индекс строки {max(row)} превышает размер {shape[0]}")
        if col and max(col) >= shape[1]:
            raise ValueError(f"Индекс столбца {max(col)} превышает размер {shape[1]}")

    def to_dense(self) -> List[List[float]]:
        """Преобразует COO в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]
        
        for i in range(self.nnz):
            dense[self.row[i]][self.col[i]] = self.data[i]
        
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц."""
        if self.shape != other.shape:
            raise ValueError("Размерности матриц не совпадают")
        
        # Преобразуем other в COO если нужно
        if not isinstance(other, COOMatrix):
            other_dense = other.to_dense()
            other_coo = COOMatrix.from_dense(other_dense)
        else:
            other_coo = other
        
        # Создаем словарь для суммирования
        result_dict = {}
        
        # Добавляем элементы из первой матрицы
        for i in range(self.nnz):
            key = (self.row[i], self.col[i])
            result_dict[key] = self.data[i]
        
        # Добавляем элементы из второй матрицы
        for i in range(other_coo.nnz):
            key = (other_coo.row[i], other_coo.col[i])
            result_dict[key] = result_dict.get(key, 0.0) + other_coo.data[i]
        
        # Удаляем нулевые элементы и собираем результат
        result_data = []
        result_row = []
        result_col = []
        
        for (r, c), val in sorted(result_dict.items()):
            if abs(val) > TOL:
                result_data.append(val)
                result_row.append(r)
                result_col.append(c)
        
        return COOMatrix(result_data, result_row, result_col, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        if abs(scalar) < TOL:
            return COOMatrix([], [], [], self.shape)
        
        new_data = []
        new_row = []
        new_col = []
        
        for i in range(self.nnz):
            val = self.data[i] * scalar
            if abs(val) > TOL:
                new_data.append(val)
                new_row.append(self.row[i])
                new_col.append(self.col[i])
        
        return COOMatrix(new_data, new_row, new_col, self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        from CSR import CSRMatrix
        from CSC import CSCMatrix
        
        # Создаем транспонированную COO
        new_shape = (self.shape[1], self.shape[0])
        return COOMatrix(self.data.copy(), self.col.copy(), self.row.copy(), new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""
        if self.shape[1] != other.shape[0]:
            raise ValueError("Несовместимые размерности для умножения")
        
        # Преобразуем в плотные для умножения
        dense_self = self.to_dense()
        dense_other = other.to_dense()
        
        rows_A, cols_A = self.shape
        rows_B, cols_B = other.shape
        
        # Умножение матриц
        result_data = []
        result_row = []
        result_col = []
        
        for i in range(rows_A):
            for j in range(cols_B):
                val = 0.0
                for k in range(cols_A):
                    val += dense_self[i][k] * dense_other[k][j]
                if abs(val) > TOL:
                    result_data.append(val)
                    result_row.append(i)
                    result_col.append(j)
        
        return COOMatrix(result_data, result_row, result_col, (rows_A, cols_B))

    @classmethod
    def from_dense(cls, dense_matrix: List[List[float]]) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        data = []
        rows = []
        cols = []
        
        for i, row in enumerate(dense_matrix):
            for j, val in enumerate(row):
                if abs(val) > TOL:
                    data.append(float(val))
                    rows.append(i)
                    cols.append(j)
        
        shape = (len(dense_matrix), len(dense_matrix[0]) if dense_matrix else 0)
        return cls(data, rows, cols, shape)

    def _to_csc(self) -> 'CSCMatrix':
        """Преобразование COOMatrix в CSCMatrix."""
        from CSC import CSCMatrix
        
        if self.nnz == 0:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)
        
        # Сортируем по столбцам, затем по строкам
        sorted_indices = sorted(range(self.nnz), 
                               key=lambda i: (self.col[i], self.row[i]))
        
        data = [self.data[i] for i in sorted_indices]
        indices = [self.row[i] for i in sorted_indices]
        
        # Строим indptr
        cols = self.shape[1]
        indptr = [0] * (cols + 1)
        
        for col in self.col:
            indptr[col + 1] += 1
        
        for i in range(1, cols + 1):
            indptr[i] += indptr[i - 1]
        
        return CSCMatrix(data, indices, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """Преобразование COOMatrix в CSRMatrix."""
        from CSR import CSRMatrix
        
        if self.nnz == 0:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)
        
        # Сортируем по строкам, затем по столбцам
        sorted_indices = sorted(range(self.nnz), 
                               key=lambda i: (self.row[i], self.col[i]))
        
        data = [self.data[i] for i in sorted_indices]
        indices = [self.col[i] for i in sorted_indices]
        
        # Строим indptr
        rows = self.shape[0]
        indptr = [0] * (rows + 1)
        
        for row in self.row:
            indptr[row + 1] += 1
        
        for i in range(1, rows + 1):
            indptr[i] += indptr[i - 1]
        
        return CSRMatrix(data, indices, indptr, self.shape)
