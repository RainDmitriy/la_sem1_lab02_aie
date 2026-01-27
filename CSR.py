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
            for k in range(start, end):
                col = self.indices[k]
                dense[i][col] = self.data[k]
        
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        if isinstance(other, CSRMatrix):
            # Сложение двух CSR матриц
            from COO import COOMatrix
            
            # Преобразуем в COO, складываем, преобразуем обратно
            coo_self = self._to_coo()
            coo_other = other._to_coo()
            coo_result = coo_self._add_impl(coo_other)
            return coo_result._to_csr()
        else:
            # Для других типов преобразуем в COO
            from COO import COOMatrix
            
            coo_self = self._to_coo()
            return coo_self._add_impl(other)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        if abs(scalar) < 1e-10:
            return CSRMatrix([], [], [0] * (self.rows + 1), self.shape)
        new_data = [val * scalar for val in self.data]
        return CSRMatrix(new_data, self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Возвращает CSC матрицу.
        """
        from CSC import CSCMatrix
        
        rows, cols = self.shape
        
        if self.nnz == 0:
            return CSCMatrix([], [], [0] * (cols + 1), (cols, rows))
        
        # Создаем временные структуры для транспонирования
        col_counts = [0] * cols
        for j in self.indices:
            col_counts[j] += 1
        
        # Строим indptr для CSC
        csc_indptr = [0] * (cols + 1)
        for j in range(cols):
            csc_indptr[j + 1] = csc_indptr[j] + col_counts[j]
        
        # Рабочий массив для текущей позиции в каждом столбце
        current_pos = csc_indptr[:]
        
        # Массивы для CSC
        csc_data = [0.0] * self.nnz
        csc_indices = [0] * self.nnz
        
        # Заполняем CSC
        for i in range(rows):
            for k in range(self.indptr[i], self.indptr[i + 1]):
                j = self.indices[k]
                val = self.data[k]
                
                pos = current_pos[j]
                csc_data[pos] = val
                csc_indices[pos] = i
                current_pos[j] += 1
        
        return CSCMatrix(csc_data, csc_indices, csc_indptr, (cols, rows))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        from CSC import CSCMatrix
        from COO import COOMatrix
        
        result_rows = self.rows
        result_cols = other.cols
        
        if isinstance(other, CSCMatrix):
            # CSR * CSC - эффективный алгоритм
            result_data = []
            result_indices = []
            result_indptr = [0]
            
            for i in range(result_rows):
                # Вектор для накопления результатов строки
                row_result = {}
                
                # Ненулевые элементы в строке i текущей матрицы
                row_start = self.indptr[i]
                row_end = self.indptr[i + 1]
                
                for k in range(row_start, row_end):
                    col_a = self.indices[k]
                    val_a = self.data[k]
                    
                    # Добавляем вклад от умножения на столбец col_a матрицы other
                    col_start = other.indptr[col_a]
                    col_end = other.indptr[col_a + 1]
                    
                    for l in range(col_start, col_end):
                        j = other.indices[l]
                        val_b = other.data[l]
                        row_result[j] = row_result.get(j, 0.0) + val_a * val_b
                
                # Сортируем индексы столбцов и добавляем ненулевые элементы
                sorted_cols = sorted(row_result.keys())
                row_nnz = 0
                for j in sorted_cols:
                    val = row_result[j]
                    if abs(val) > 1e-10:
                        result_data.append(val)
                        result_indices.append(j)
                        row_nnz += 1
                
                result_indptr.append(result_indptr[-1] + row_nnz)
            
            return CSRMatrix(result_data, result_indices, result_indptr, (result_rows, result_cols))
        else:
            # Общий случай - через плотные матрицы
            dense_self = self.to_dense()
            dense_other = other.to_dense()
            result = [[0.0] * result_cols for _ in range(result_rows)]
            
            for i in range(result_rows):
                for k in range(self.cols):
                    val = dense_self[i][k]
                    if abs(val) > 1e-10:
                        for j in range(result_cols):
                            result[i][j] += val * dense_other[k][j]
            
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
                if abs(val) > 1e-10:
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
        
        # Подсчитываем количество элементов в каждом столбце
        col_counts = [0] * cols
        for j in self.indices:
            col_counts[j] += 1
        
        # Строим indptr для CSC
        csc_indptr = [0] * (cols + 1)
        for j in range(cols):
            csc_indptr[j + 1] = csc_indptr[j] + col_counts[j]
        
        # Рабочий массив для текущей позиции в каждом столбце
        current_pos = csc_indptr[:]
        
        # Массивы для CSC
        csc_data = [0.0] * self.nnz
        csc_indices = [0] * self.nnz
        
        # Заполняем CSC
        for i in range(rows):
            for k in range(self.indptr[i], self.indptr[i + 1]):
                j = self.indices[k]
                val = self.data[k]
                
                pos = current_pos[j]
                csc_data[pos] = val
                csc_indices[pos] = i
                current_pos[j] += 1
        
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
