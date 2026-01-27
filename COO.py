from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.row = row
        self.col = col
        self.nnz = len(data)

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]
        
        for i in range(self.nnz):
            r = self.row[i]
            c = self.col[i]
            dense[r][c] = self.data[i]
        
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц."""
        if self.shape != other.shape:
            raise ValueError("Размерности матриц не совпадают")
        
        # Преобразуем вторую матрицу в COO если нужно
        if not isinstance(other, COOMatrix):
            other_coo = other._to_coo() if hasattr(other, '_to_coo') else COOMatrix.from_dense(other.to_dense())
        else:
            other_coo = other
        
        # Создаем словари для быстрого поиска
        self_dict = {}
        for i in range(self.nnz):
            key = (self.row[i], self.col[i])
            self_dict[key] = self.data[i]
        
        other_dict = {}
        for i in range(other_coo.nnz):
            key = (other_coo.row[i], other_coo.col[i])
            other_dict[key] = other_coo.data[i]
        
        # Объединяем ключи
        all_keys = set(self_dict.keys()) | set(other_dict.keys())
        
        result_data = []
        result_row = []
        result_col = []
        
        for key in sorted(all_keys):
            r, c = key
            val = self_dict.get(key, 0.0) + other_dict.get(key, 0.0)
            if val != 0:
                result_data.append(val)
                result_row.append(r)
                result_col.append(c)
        
        return COOMatrix(result_data, result_row, result_col, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        if scalar == 0:
            # Возвращаем нулевую матрицу
            return COOMatrix([], [], [], self.shape)
        
        new_data = [val * scalar for val in self.data]
        return COOMatrix(new_data, self.row.copy(), self.col.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        # Меняем местами строки и столбцы
        new_shape = (self.shape[1], self.shape[0])
        return COOMatrix(self.data.copy(), self.col.copy(), self.row.copy(), new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""
        if self.shape[1] != other.shape[0]:
            raise ValueError("Несовместимые размерности для умножения")
        
        # Преобразуем в CSR для эффективного умножения
        csr_self = self._to_csr()
        
        # Преобразуем other в CSR если нужно
        if not isinstance(other, type(csr_self)):
            if hasattr(other, '_to_csr'):
                other_csr = other._to_csr()
            else:
                other_csr = type(csr_self).from_dense(other.to_dense())
        else:
            other_csr = other
        
        return csr_self._matmul_impl(other_csr)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        data = []
        rows = []
        cols = []
        
        for i, row in enumerate(dense_matrix):
            for j, val in enumerate(row):
                if val != 0:
                    data.append(val)
                    rows.append(i)
                    cols.append(j)
        
        shape = (len(dense_matrix), len(dense_matrix[0]) if dense_matrix else 0)
        return cls(data, rows, cols, shape)

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix
        
        if self.nnz == 0:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)
        
        # Сортируем по столбцам, затем по строкам
        sorted_indices = sorted(range(self.nnz), 
                               key=lambda i: (self.col[i], self.row[i]))
        
        data = [self.data[i] for i in sorted_indices]
        indices = [self.row[i] for i in sorted_indices]
        
        # Строим indptr для CSC
        indptr = [0] * (self.shape[1] + 1)
        for col in self.col:
            indptr[col + 1] += 1
        
        # Накопление
        for i in range(1, len(indptr)):
            indptr[i] += indptr[i - 1]
        
        return CSCMatrix(data, indices, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix
        
        if self.nnz == 0:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)
        
        # Сортируем по строкам, затем по столбцам
        sorted_indices = sorted(range(self.nnz), 
                               key=lambda i: (self.row[i], self.col[i]))
        
        data = [self.data[i] for i in sorted_indices]
        indices = [self.col[i] for i in sorted_indices]
        
        # Строим indptr для CSR
        indptr = [0] * (self.shape[0] + 1)
        for row in self.row:
            indptr[row + 1] += 1
        
        # Накопление
        for i in range(1, len(indptr)):
            indptr[i] += indptr[i - 1]
        
        return CSRMatrix(data, indices, indptr, self.shape)
