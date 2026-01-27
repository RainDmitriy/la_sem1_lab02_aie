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
            dense[self.row[i]][self.col[i]] = self.data[i]
        
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц."""
        if self.shape != other.shape:
            raise ValueError("Размерности матриц не совпадают")
        
        # Если other тоже COO
        if isinstance(other, COOMatrix):
            # Создаем словари
            dict1 = {}
            for i in range(self.nnz):
                key = (self.row[i], self.col[i])
                dict1[key] = self.data[i]
            
            dict2 = {}
            for i in range(other.nnz):
                key = (other.row[i], other.col[i])
                dict2[key] = other.data[i]
            
            # Объединяем
            result_data = []
            result_row = []
            result_col = []
            
            all_keys = set(dict1.keys()) | set(dict2.keys())
            for r, c in sorted(all_keys):
                val = dict1.get((r, c), 0.0) + dict2.get((r, c), 0.0)
                # Убираем порог - храним все, что не точно 0
                if val != 0.0:
                    result_data.append(val)
                    result_row.append(r)
                    result_col.append(c)
            
            return COOMatrix(result_data, result_row, result_col, self.shape)
        else:
            # Иначе преобразуем в плотные
            dense_self = self.to_dense()
            dense_other = other.to_dense()
            rows, cols = self.shape
            
            result_data = []
            result_row = []
            result_col = []
            
            for i in range(rows):
                for j in range(cols):
                    val = dense_self[i][j] + dense_other[i][j]
                    if val != 0.0:
                        result_data.append(val)
                        result_row.append(i)
                        result_col.append(j)
            
            return COOMatrix(result_data, result_row, result_col, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        if scalar == 0.0:
            return COOMatrix([], [], [], self.shape)
        
        new_data = [val * scalar for val in self.data]
        return COOMatrix(new_data, self.row.copy(), self.col.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        new_shape = (self.shape[1], self.shape[0])
        return COOMatrix(self.data.copy(), self.col.copy(), self.row.copy(), new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""
        if self.shape[1] != other.shape[0]:
            raise ValueError("Несовместимые размерности для умножения")
        
        # Преобразуем self в плотный формат для простоты
        dense_self = self.to_dense()
        
        if isinstance(other, COOMatrix):
            dense_other = other.to_dense()
        else:
            dense_other = other.to_dense()
        
        rows_A, cols_A = self.shape
        rows_B, cols_B = other.shape
        
        result_data = []
        result_row = []
        result_col = []
        
        for i in range(rows_A):
            for j in range(cols_B):
                val = 0.0
                for k in range(cols_A):
                    val += dense_self[i][k] * dense_other[k][j]
                if val != 0.0:
                    result_data.append(val)
                    result_row.append(i)
                    result_col.append(j)
        
        return COOMatrix(result_data, result_row, result_col, (rows_A, cols_B))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        data = []
        rows = []
        cols = []
        
        for i, row in enumerate(dense_matrix):
            for j, val in enumerate(row):
                if val != 0.0:
                    data.append(float(val))
                    rows.append(i)
                    cols.append(j)
        
        shape = (len(dense_matrix), len(dense_matrix[0]) if dense_matrix else 0)
        return cls(data, rows, cols, shape)

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        if self.nnz == 0:
            from CSC import CSCMatrix
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
        
        from CSC import CSCMatrix
        return CSCMatrix(data, indices, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        if self.nnz == 0:
            from CSR import CSRMatrix
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
        
        from CSR import CSRMatrix
        return CSRMatrix(data, indices, indptr, self.shape)
