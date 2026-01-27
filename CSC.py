from base import Matrix
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.nnz = len(data)

    def to_dense(self) -> DenseMatrix:
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]
        
        for j in range(cols):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for k in range(start, end):
                row = self.indices[k]
                dense[row][j] = self.data[k]
        
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        if isinstance(other, CSCMatrix):
            # Сложение двух CSC матриц
            from COO import COOMatrix
            
            # Преобразуем в COO, складываем, преобразуем обратно
            coo_self = self._to_coo()
            coo_other = other._to_coo()
            coo_result = coo_self._add_impl(coo_other)
            return coo_result._to_csc()
        else:
            # Для других типов преобразуем в COO
            from COO import COOMatrix
            
            coo_self = self._to_coo()
            return coo_self._add_impl(other)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        if abs(scalar) < 1e-10:
            return CSCMatrix([], [], [0] * (self.cols + 1), self.shape)
        new_data = [val * scalar for val in self.data]
        return CSCMatrix(new_data, self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Возвращает CSR матрицу.
        """
        from CSR import CSRMatrix
        
        rows, cols = self.shape
        
        if self.nnz == 0:
            return CSRMatrix([], [], [0] * (rows + 1), (cols, rows))
        
        # Создаем временные структуры для транспонирования
        row_counts = [0] * rows
        for i in self.indices:
            row_counts[i] += 1
        
        # Строим indptr для CSR
        csr_indptr = [0] * (rows + 1)
        for i in range(rows):
            csr_indptr[i + 1] = csr_indptr[i] + row_counts[i]
        
        # Рабочий массив для текущей позиции в каждой строке
        current_pos = csr_indptr[:]
        
        # Массивы для CSR
        csr_data = [0.0] * self.nnz
        csr_indices = [0] * self.nnz
        
        # Заполняем CSR
        for j in range(cols):
            for k in range(self.indptr[j], self.indptr[j + 1]):
                i = self.indices[k]
                val = self.data[k]
                
                pos = current_pos[i]
                csr_data[pos] = val
                csr_indices[pos] = j
                current_pos[i] += 1
        
        return CSRMatrix(csr_data, csr_indices, csr_indptr, (cols, rows))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        from CSR import CSRMatrix
        
        result_rows = self.rows
        result_cols = other.cols
        
        if isinstance(other, CSCMatrix):
            # CSC * CSC - преобразуем в CSR для умножения
            csr_self = self._to_csr()
            return csr_self._matmul_impl(other)
        elif isinstance(other, CSRMatrix):
            # CSC * CSR
            # Преобразуем текущую матрицу в CSR
            csr_self = self._to_csr()
            return csr_self._matmul_impl(other)
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
            
            return CSCMatrix.from_dense(result)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        
        data = []
        indices = []
        indptr = [0]
        
        for j in range(cols):
            nnz_in_col = 0
            for i in range(rows):
                val = dense_matrix[i][j]
                if abs(val) > 1e-10:
                    data.append(val)
                    indices.append(i)
                    nnz_in_col += 1
            indptr.append(indptr[-1] + nnz_in_col)
        
        return cls(data, indices, indptr, (rows, cols))

    def _to_csr(self) -> 'CSRMatrix':
        from CSR import CSRMatrix
        
        rows, cols = self.shape
        
        if self.nnz == 0:
            return CSRMatrix([], [], [0] * (rows + 1), self.shape)
        
        # Подсчитываем количество элементов в каждой строке
        row_counts = [0] * rows
        for i in self.indices:
            row_counts[i] += 1
        
        # Строим indptr для CSR
        csr_indptr = [0] * (rows + 1)
        for i in range(rows):
            csr_indptr[i + 1] = csr_indptr[i] + row_counts[i]
        
        # Рабочий массив для текущей позиции в каждой строке
        current_pos = csr_indptr[:]
        
        # Массивы для CSR
        csr_data = [0.0] * self.nnz
        csr_indices = [0] * self.nnz
        
        # Заполняем CSR
        for j in range(cols):
            for k in range(self.indptr[j], self.indptr[j + 1]):
                i = self.indices[k]
                val = self.data[k]
                
                pos = current_pos[i]
                csr_data[pos] = val
                csr_indices[pos] = j
                current_pos[i] += 1
        
        return CSRMatrix(csr_data, csr_indices, csr_indptr, (rows, cols))

    def _to_coo(self) -> 'COOMatrix':
        from COO import COOMatrix
        
        data = self.data[:]
        rows = self.indices[:]
        cols = []
        
        for j in range(self.cols):
            for k in range(self.indptr[j], self.indptr[j + 1]):
                cols.append(j)
        
        return COOMatrix(data, rows, cols, self.shape)
