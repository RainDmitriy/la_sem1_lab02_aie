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
        """Преобразует CSR в плотную матрицу."""
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
        if self.shape != other.shape:
            raise ValueError("Размерности матриц не совпадают")
        
        # Если other не CSR, преобразуем его в CSR
        if not isinstance(other, CSRMatrix):
            if hasattr(other, '_to_csr'):
                other_csr = other._to_csr()
            else:
                other_csr = CSRMatrix.from_dense(other.to_dense())
        else:
            other_csr = other
        
        rows, cols = self.shape
        result_data = []
        result_indices = []
        result_indptr = [0]
        
        for i in range(rows):
            # Получаем элементы текущей строки из обеих матриц
            self_start = self.indptr[i]
            self_end = self.indptr[i + 1]
            other_start = other_csr.indptr[i]
            other_end = other_csr.indptr[i + 1]
            
            idx_self = self_start
            idx_other = other_start
            
            # Слияние двух отсортированных списков (по столбцам)
            while idx_self < self_end and idx_other < other_end:
                col_self = self.indices[idx_self]
                col_other = other_csr.indices[idx_other]
                
                if col_self < col_other:
                    result_data.append(self.data[idx_self])
                    result_indices.append(col_self)
                    idx_self += 1
                elif col_self > col_other:
                    result_data.append(other_csr.data[idx_other])
                    result_indices.append(col_other)
                    idx_other += 1
                else:  # col_self == col_other
                    val = self.data[idx_self] + other_csr.data[idx_other]
                    if val != 0:
                        result_data.append(val)
                        result_indices.append(col_self)
                    idx_self += 1
                    idx_other += 1
            
            # Добавляем оставшиеся элементы из первой матрицы
            while idx_self < self_end:
                result_data.append(self.data[idx_self])
                result_indices.append(self.indices[idx_self])
                idx_self += 1
            
            # Добавляем оставшиеся элементы из второй матрицы
            while idx_other < other_end:
                result_data.append(other_csr.data[idx_other])
                result_indices.append(other_csr.indices[idx_other])
                idx_other += 1
            
            result_indptr.append(len(result_data))
        
        return CSRMatrix(result_data, result_indices, result_indptr, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        if scalar == 0:
            # Возвращаем нулевую матрицу
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)
        
        new_data = [val * scalar for val in self.data]
        return CSRMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Результат - в CSC формате.
        """
        return self._to_csc()

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        if self.shape[1] != other.shape[0]:
            raise ValueError("Несовместимые размерности для умножения")
        
        rows_A, cols_A = self.shape
        rows_B, cols_B = other.shape
        
        # Если other не CSR, преобразуем в CSR
        if not isinstance(other, CSRMatrix):
            if hasattr(other, '_to_csr'):
                other_csr = other._to_csr()
            else:
                other_csr = CSRMatrix.from_dense(other.to_dense())
        else:
            other_csr = other
        
        # Алгоритм умножения CSR матриц
        result_data = []
        result_indices = []
        result_indptr = [0]
        
        # Временный массив для накопления результатов строки
        temp = [0.0] * cols_B
        
        for i in range(rows_A):
            # Обнуляем временный массив
            for j in range(cols_B):
                temp[j] = 0.0
            
            # Умножаем строку i матрицы A на матрицу B
            for k_idx in range(self.indptr[i], self.indptr[i + 1]):
                k = self.indices[k_idx]
                a_val = self.data[k_idx]
                
                # Добавляем вклад от строки k матрицы B
                for b_idx in range(other_csr.indptr[k], other_csr.indptr[k + 1]):
                    j = other_csr.indices[b_idx]
                    temp[j] += a_val * other_csr.data[b_idx]
            
            # Сохраняем ненулевые элементы строки результата
            row_nnz = 0
            for j in range(cols_B):
                if temp[j] != 0:
                    result_data.append(temp[j])
                    result_indices.append(j)
                    row_nnz += 1
            
            result_indptr.append(result_indptr[-1] + row_nnz)
        
        return CSRMatrix(result_data, result_indices, result_indptr, (rows_A, cols_B))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        if not dense_matrix:
            return cls([], [], [0], (0, 0))
        
        rows = len(dense_matrix)
        cols = len(dense_matrix[0])
        
        data = []
        indices = []
        indptr = [0]
        
        for i in range(rows):
            nnz_in_row = 0
            for j in range(cols):
                val = dense_matrix[i][j]
                if val != 0:
                    data.append(val)
                    indices.append(j)
                    nnz_in_row += 1
            indptr.append(indptr[-1] + nnz_in_row)
        
        return cls(data, indices, indptr, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSRMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix
        
        rows, cols = self.shape
        
        if self.nnz == 0:
            return CSCMatrix([], [], [0] * (cols + 1), self.shape)
        
        # Подсчет количества ненулевых элементов в каждом столбце
        col_counts = [0] * cols
        for col in self.indices:
            col_counts[col] += 1
        
        # Строим indptr для CSC
        indptr = [0] * (cols + 1)
        for j in range(cols):
            indptr[j + 1] = indptr[j] + col_counts[j]
        
        # Рабочие массивы для построения
        current_pos = indptr.copy()
        new_data = [0.0] * self.nnz
        new_indices = [0] * self.nnz
        
        # Заполняем CSC данные
        for i in range(rows):
            for k in range(self.indptr[i], self.indptr[i + 1]):
                j = self.indices[k]
                pos = current_pos[j]
                new_data[pos] = self.data[k]
                new_indices[pos] = i
                current_pos[j] += 1
        
        return CSCMatrix(new_data, new_indices, indptr, self.shape)
    
    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSRMatrix в COOMatrix.
        """
        from COO import COOMatrix
        
        if self.nnz == 0:
            return COOMatrix([], [], [], self.shape)
        
        data = []
        rows = []
        cols = []
        
        for i in range(self.shape[0]):
            for k in range(self.indptr[i], self.indptr[i + 1]):
                data.append(self.data[k])
                rows.append(i)
                cols.append(self.indices[k])
        
        return COOMatrix(data, rows, cols, self.shape)
