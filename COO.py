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
        for val, r, c in zip(self.data, self.row, self.col):
            dense[r][c] = val
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц."""
        from CSR import CSRMatrix
        from CSC import CSCMatrix
        
        if isinstance(other, (CSRMatrix, CSCMatrix)):
            other = other._to_coo()
        
        result_data = []
        result_row = []
        result_col = []
        
        # Создаём словарь для быстрого доступа к значениям текущей матрицы
        current = {}
        for i in range(self.nnz):
            key = (self.row[i], self.col[i])
            current[key] = self.data[i]
        
        # Обрабатываем матрицу other
        if isinstance(other, COOMatrix):
            other_dict = {}
            for i in range(other.nnz):
                key = (other.row[i], other.col[i])
                other_dict[key] = other.data[i]
        else:
            other_dense = other.to_dense()
            other_dict = {}
            rows, cols = other.shape
            for r in range(rows):
                for c in range(cols):
                    if other_dense[r][c] != 0:
                        other_dict[(r, c)] = other_dense[r][c]
        
        # Объединяем значения
        all_keys = set(current.keys()) | set(other_dict.keys())
        for r, c in sorted(all_keys):
            val = current.get((r, c), 0) + other_dict.get((r, c), 0)
            if val != 0:
                result_data.append(val)
                result_row.append(r)
                result_col.append(c)
        
        return COOMatrix(result_data, result_row, result_col, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        if scalar == 0:
            return COOMatrix([], [], [], self.shape)
        new_data = [val * scalar for val in self.data]
        return COOMatrix(new_data, self.row[:], self.col[:], self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        return COOMatrix(self.data[:], self.col[:], self.row[:], (self.cols, self.rows))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""
        # Преобразуем в CSR для эффективного умножения
        csr_self = self._to_csr()
        return csr_self._matmul_impl(other)

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
                if val != 0:
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
        sorted_indices = sorted(range(self.nnz), key=lambda i: (self.col[i], self.row[i]))
        
        data = [self.data[i] for i in sorted_indices]
        indices = [self.row[i] for i in sorted_indices]
        
        # Строим indptr
        indptr = [0] * (self.cols + 1)
        for i in range(self.nnz):
            col_idx = self.col[sorted_indices[i]]
            indptr[col_idx + 1] += 1
        
        for j in range(self.cols):
            indptr[j + 1] += indptr[j]
        
        return CSCMatrix(data, indices, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix
        
        if self.nnz == 0:
            return CSRMatrix([], [], [0] * (self.rows + 1), self.shape)
        
        # Сортируем по строкам, затем по столбцам
        sorted_indices = sorted(range(self.nnz), key=lambda i: (self.row[i], self.col[i]))
        
        data = [self.data[i] for i in sorted_indices]
        indices = [self.col[i] for i in sorted_indices]
        
        # Строим indptr
        indptr = [0] * (self.rows + 1)
        for i in range(self.nnz):
            row_idx = self.row[sorted_indices[i]]
            indptr[row_idx + 1] += 1
        
        for i in range(self.rows):
            indptr[i + 1] += indptr[i]
        
        return CSRMatrix(data, indices, indptr, self.shape)
