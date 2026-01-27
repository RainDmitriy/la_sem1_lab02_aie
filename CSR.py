from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.nnz = len(data)

    def to_dense(self) -> DenseMatrix:
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]
        
        for i in range(rows):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for idx in range(start, end):
                j = self.indices[idx]
                dense[i][j] = self.data[idx]
        
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        # Преобразуем в COO для сложения
        from COO import COOMatrix
        
        coo_self = self._to_coo()
        coo_other = other._to_coo()
        coo_result = coo_self._add_impl(coo_other)
        return coo_result._to_csr()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        if scalar == 0.0:
            return CSRMatrix([], [], [0] * (self.rows + 1), self.shape)
        
        new_data = [val * scalar for val in self.data]
        return CSRMatrix(new_data, self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        from CSC import CSCMatrix
        
        rows, cols = self.shape
        
        if self.nnz == 0:
            return CSCMatrix([], [], [0] * (cols + 1), (cols, rows))
        
        # Подсчитываем ненулевые элементы в каждом столбце
        col_counts = [0] * cols
        for j in self.indices:
            col_counts[j] += 1
        
        # Строим indptr для CSC
        csc_indptr = [0] * (cols + 1)
        for j in range(cols):
            csc_indptr[j + 1] = csc_indptr[j] + col_counts[j]
        
        # Временный массив для позиций в каждом столбце
        temp_pos = csc_indptr[:]
        
        # Создаем массивы для CSC
        csc_data = [0.0] * self.nnz
        csc_indices = [0] * self.nnz
        
        # Заполняем CSC
        for i in range(rows):
            for k in range(self.indptr[i], self.indptr[i + 1]):
                j = self.indices[k]
                pos = temp_pos[j]
                csc_data[pos] = self.data[k]
                csc_indices[pos] = i
                temp_pos[j] += 1
        
        return CSCMatrix(csc_data, csc_indices, csc_indptr, (cols, rows))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        # Используем алгоритм умножения разреженных матриц
        result_rows = self.rows
        result_cols = other.cols
        
        # Если other - CSC, используем оптимизированный алгоритм
        from CSC import CSCMatrix
        if isinstance(other, CSCMatrix):
            # CSR * CSC
            result_data = []
            result_indices = []
            result_indptr = [0]
            
            for i in range(result_rows):
                # Создаем временный массив для строки результата
                temp_row = [0.0] * result_cols
                
                # Получаем строку i из CSR
                row_start = self.indptr[i]
                row_end = self.indptr[i + 1]
                
                for k in range(row_start, row_end):
                    col_a = self.indices[k]
                    val_a = self.data[k]
                    
                    # Добавляем вклад от столбца col_a матрицы other
                    col_start = other.indptr[col_a]
                    col_end = other.indptr[col_a + 1]
                    
                    for l in range(col_start, col_end):
                        j = other.indices[l]
                        temp_row[j] += val_a * other.data[l]
                
                # Формируем CSR строку результата
                row_nnz = 0
                for j in range(result_cols):
                    if abs(temp_row[j]) > 1e-12:
                        result_data.append(temp_row[j])
                        result_indices.append(j)
                        row_nnz += 1
                
                result_indptr.append(result_indptr[-1] + row_nnz)
            
            return CSRMatrix(result_data, result_indices, result_indptr, (result_rows, result_cols))
        else:
            # Общий случай
            dense_other = other.to_dense()
            result = [[0.0] * result_cols for _ in range(result_rows)]
            
            for i in range(result_rows):
                row_start = self.indptr[i]
                row_end = self.indptr[i + 1]
                
                for k in range(row_start, row_end):
                    col_a = self.indices[k]
                    val_a = self.data[k]
                    
                    for j in range(result_cols):
                        result[i][j] += val_a * dense_other[col_a][j]
            
            return CSRMatrix.from_dense(result)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        
        data = []
        indices = []
        indptr = [0]
        
        for i in range(rows):
            nnz_in_row = 0
            for j in range(cols):
                val = dense_matrix[i][j]
                if abs(val) > 1e-12:
                    data.append(val)
                    indices.append(j)
                    nnz_in_row += 1
            indptr.append(indptr[-1] + nnz_in_row)
        
        return cls(data, indices, indptr, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        from CSC import CSCMatrix
        
        rows, cols = self.shape
        
        if self.nnz == 0:
            return CSCMatrix([], [], [0] * (cols + 1), self.shape)
        
        # Подсчитываем ненулевые элементы в каждом столбце
        col_counts = [0] * cols
        for j in self.indices:
            col_counts[j] += 1
        
        # Строим indptr для CSC
        csc_indptr = [0] * (cols + 1)
        for j in range(cols):
            csc_indptr[j + 1] = csc_indptr[j] + col_counts[j]
        
        # Временный массив для позиций
        temp_pos = csc_indptr[:]
        
        # Создаем массивы для CSC
        csc_data = [0.0] * self.nnz
        csc_indices = [0] * self.nnz
        
        # Заполняем CSC
        for i in range(rows):
            for k in range(self.indptr[i], self.indptr[i + 1]):
                j = self.indices[k]
                pos = temp_pos[j]
                csc_data[pos] = self.data[k]
                csc_indices[pos] = i
                temp_pos[j] += 1
        
        return CSCMatrix(csc_data, csc_indices, csc_indptr, self.shape)

    def _to_coo(self) -> 'COOMatrix':
        from COO import COOMatrix
        
        data = self.data[:]
        rows = []
        cols = self.indices[:]
        
        for i in range(self.rows):
            for k in range(self.indptr[i], self.indptr[i + 1]):
                rows.append(i)
        
        return COOMatrix(data, rows, cols, self.shape)
