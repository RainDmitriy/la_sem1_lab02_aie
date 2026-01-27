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
        # Преобразуем в COO для сложения
        from COO import COOMatrix
        
        coo_self = self._to_coo()
        coo_other = other._to_coo()
        coo_result = coo_self._add_impl(coo_other)
        return coo_result._to_csr()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        if abs(scalar) < 1e-7:
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
        
        # Считаем количество элементов в каждом столбце
        col_counts = [0] * cols
        for j in self.indices:
            col_counts[j] += 1
        
        # Строим indptr для CSC
        csc_indptr = [0] * (cols + 1)
        for j in range(cols):
            csc_indptr[j + 1] = csc_indptr[j] + col_counts[j]
        
        # Временный массив для текущих позиций
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
        """Умножение CSR матриц."""
        # Обработка разных типов матриц
        from CSC import CSCMatrix
        from COO import COOMatrix
        
        if isinstance(other, CSCMatrix):
            # CSR * CSC - оптимальный случай
            return self._multiply_csr_csc(other)
        elif isinstance(other, COOMatrix):
            # CSR * COO
            other_csc = other._to_csc()
            return self._multiply_csr_csc(other_csc)
        elif isinstance(other, CSRMatrix):
            # CSR * CSR
            other_csc = other._to_csc()
            return self._multiply_csr_csc(other_csc)
        else:
            # Общий случай
            dense_result = self._dense_matmul(other)
            return CSRMatrix.from_dense(dense_result)

    def _multiply_csr_csc(self, csc: 'CSCMatrix') -> 'CSRMatrix':
        """Умножение CSR на CSC."""
        rows = self.rows
        cols = csc.cols
        
        result_data = []
        result_indices = []
        result_indptr = [0]
        
        # Временный массив для строки результата
        temp_row = [0.0] * cols
        
        for i in range(rows):
            # Обнуляем временный массив
            for j in range(cols):
                temp_row[j] = 0.0
            
            # Умножаем строку i на все столбцы
            for k in range(self.indptr[i], self.indptr[i + 1]):
                col_a = self.indices[k]
                val_a = self.data[k]
                
                # Добавляем вклад от столбца col_a матрицы csc
                for l in range(csc.indptr[col_a], csc.indptr[col_a + 1]):
                    j = csc.indices[l]
                    temp_row[j] += val_a * csc.data[l]
            
            # Формируем CSR строку
            row_nnz = 0
            for j in range(cols):
                if abs(temp_row[j]) > 1e-7:
                    result_data.append(temp_row[j])
                    result_indices.append(j)
                    row_nnz += 1
            
            result_indptr.append(result_indptr[-1] + row_nnz)
        
        return CSRMatrix(result_data, result_indices, result_indptr, (rows, cols))

    def _dense_matmul(self, other: 'Matrix') -> DenseMatrix:
        """Умножение через плотные матрицы."""
        dense_self = self.to_dense()
        dense_other = other.to_dense()
        rows, cols = self.rows, other.cols
        result = [[0.0] * cols for _ in range(rows)]
        
        for i in range(rows):
            for j in range(cols):
                s = 0.0
                for k in range(self.cols):
                    s += dense_self[i][k] * dense_other[k][j]
                result[i][j] = s
        
        return result

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
                if abs(val) > 1e-7:
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
        
        # Считаем количество элементов в каждом столбце
        col_counts = [0] * cols
        for j in self.indices:
            col_counts[j] += 1
        
        # Строим indptr для CSC
        csc_indptr = [0] * (cols + 1)
        for j in range(cols):
            csc_indptr[j + 1] = csc_indptr[j] + col_counts[j]
        
        # Временный массив для текущих позиций
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
