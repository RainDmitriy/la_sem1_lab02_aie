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
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]
        for val, r, c in zip(self.data, self.row, self.col):
            dense[r][c] = val
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        if isinstance(other, COOMatrix):
            # Создаем словарь для объединения
            combined = {}
            for i in range(self.nnz):
                key = (self.row[i], self.col[i])
                combined[key] = self.data[i]
            
            for i in range(other.nnz):
                key = (other.row[i], other.col[i])
                combined[key] = combined.get(key, 0.0) + other.data[i]
            
            # Фильтруем нули
            result_data = []
            result_row = []
            result_col = []
            
            for (r, c), val in sorted(combined.items()):
                if abs(val) > 1e-12:
                    result_data.append(val)
                    result_row.append(r)
                    result_col.append(c)
            
            return COOMatrix(result_data, result_row, result_col, self.shape)
        else:
            # Для других форматов используем плотное представление
            dense_self = self.to_dense()
            dense_other = other.to_dense()
            result = [[0.0] * self.cols for _ in range(self.rows)]
            
            for i in range(self.rows):
                for j in range(self.cols):
                    result[i][j] = dense_self[i][j] + dense_other[i][j]
            
            return COOMatrix.from_dense(result)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        if scalar == 0.0:
            return COOMatrix([], [], [], self.shape)
        
        new_data = [val * scalar for val in self.data]
        return COOMatrix(new_data, self.row[:], self.col[:], self.shape)

    def transpose(self) -> 'Matrix':
        # Меняем местами строки и столбцы
        return COOMatrix(self.data[:], self.col[:], self.row[:], (self.cols, self.rows))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        # Используем алгоритм умножения разреженных матриц
        result_rows = self.rows
        result_cols = other.cols
        
        # Создаем словарь для строк текущей матрицы
        rows_dict = {}
        for i in range(self.nnz):
            r, c, val = self.row[i], self.col[i], self.data[i]
            if r not in rows_dict:
                rows_dict[r] = []
            rows_dict[r].append((c, val))
        
        # Создаем словарь для столбцов другой матрицы
        cols_dict = {}
        other_dense = other.to_dense()
        for j in range(other.cols):
            cols_dict[j] = [other_dense[i][j] for i in range(other.rows)]
        
        # Умножаем
        result_data = []
        result_row = []
        result_col = []
        
        for i in range(result_rows):
            if i in rows_dict:
                for j in range(result_cols):
                    sum_val = 0.0
                    for (k, val_ik) in rows_dict[i]:
                        sum_val += val_ik * cols_dict[j][k]
                    
                    if abs(sum_val) > 1e-12:
                        result_data.append(sum_val)
                        result_row.append(i)
                        result_col.append(j)
        
        return COOMatrix(result_data, result_row, result_col, (result_rows, result_cols))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        data = []
        row = []
        col = []
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        
        for r in range(rows):
            for c in range(cols):
                val = dense_matrix[r][c]
                if abs(val) > 1e-12:
                    data.append(val)
                    row.append(r)
                    col.append(c)
        
        return cls(data, row, col, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        from CSC import CSCMatrix
        
        if self.nnz == 0:
            return CSCMatrix([], [], [0] * (self.cols + 1), self.shape)
        
        # Сортируем по столбцам, затем по строкам
        indices = sorted(range(self.nnz), key=lambda i: (self.col[i], self.row[i]))
        
        data = [self.data[i] for i in indices]
        indices_rows = [self.row[i] for i in indices]
        
        # Строим indptr
        indptr = [0] * (self.cols + 1)
        for i in indices:
            col_idx = self.col[i]
            indptr[col_idx + 1] += 1
        
        for j in range(self.cols):
            indptr[j + 1] += indptr[j]
        
        return CSCMatrix(data, indices_rows, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        from CSR import CSRMatrix
        
        if self.nnz == 0:
            return CSRMatrix([], [], [0] * (self.rows + 1), self.shape)
        
        # Сортируем по строкам, затем по столбцам
        indices = sorted(range(self.nnz), key=lambda i: (self.row[i], self.col[i]))
        
        data = [self.data[i] for i in indices]
        indices_cols = [self.col[i] for i in indices]
        
        # Строим indptr
        indptr = [0] * (self.rows + 1)
        for i in indices:
            row_idx = self.row[i]
            indptr[row_idx + 1] += 1
        
        for i in range(self.rows):
            indptr[i + 1] += indptr[i]
        
        return CSRMatrix(data, indices_cols, indptr, self.shape)
