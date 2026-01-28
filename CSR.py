from base import Matrix
from typing import List, Tuple

TOL = 1e-12


class CSRMatrix(Matrix):
    def __init__(self, data: List[float], indices: List[int], indptr: List[int], shape: Tuple[int, int]):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.nnz = len(data)
        
        # Проверка целостности
        if len(indptr) != shape[0] + 1:
            raise ValueError(f"Длина indptr должна быть {shape[0] + 1}, получено {len(indptr)}")
        if indices and max(indices) >= shape[1]:
            raise ValueError(f"Индекс столбца {max(indices)} превышает размер {shape[1]}")

    def to_dense(self) -> List[List[float]]:
        """Преобразует CSR в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]
        
        for i in range(rows):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for idx in range(start, end):
                col = self.indices[idx]
                dense[i][col] = self.data[idx]
        
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        if self.shape != other.shape:
            raise ValueError("Размерности матриц не совпадают")
        
        # Если other тоже CSR
        if isinstance(other, CSRMatrix):
            rows, cols = self.shape
            result_data = []
            result_indices = []
            result_indptr = [0]
            
            for i in range(rows):
                self_start = self.indptr[i]
                self_end = self.indptr[i + 1]
                other_start = other.indptr[i]
                other_end = other.indptr[i + 1]
                
                idx1, idx2 = self_start, other_start
                
                while idx1 < self_end and idx2 < other_end:
                    col1 = self.indices[idx1]
                    col2 = other.indices[idx2]
                    
                    if col1 < col2:
                        result_data.append(self.data[idx1])
                        result_indices.append(col1)
                        idx1 += 1
                    elif col1 > col2:
                        result_data.append(other.data[idx2])
                        result_indices.append(col2)
                        idx2 += 1
                    else:
                        val = self.data[idx1] + other.data[idx2]
                        if abs(val) > TOL:
                            result_data.append(val)
                            result_indices.append(col1)
                        idx1 += 1
                        idx2 += 1
                
                # Добавляем оставшиеся элементы из self
                while idx1 < self_end:
                    result_data.append(self.data[idx1])
                    result_indices.append(self.indices[idx1])
                    idx1 += 1
                
                # Добавляем оставшиеся элементы из other
                while idx2 < other_end:
                    result_data.append(other.data[idx2])
                    result_indices.append(other.indices[idx2])
                    idx2 += 1
                
                result_indptr.append(len(result_data))
            
            return CSRMatrix(result_data, result_indices, result_indptr, self.shape)
        else:
            # Иначе преобразуем в плотные
            dense_self = self.to_dense()
            dense_other = other.to_dense()
            rows, cols = self.shape
            
            data = []
            indices = []
            indptr = [0]
            
            for i in range(rows):
                row_nnz = 0
                for j in range(cols):
                    val = dense_self[i][j] + dense_other[i][j]
                    if abs(val) > TOL:
                        data.append(val)
                        indices.append(j)
                        row_nnz += 1
                indptr.append(indptr[-1] + row_nnz)
            
            return CSRMatrix(data, indices, indptr, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        if scalar == 0.0:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)
        
        # Умножаем все значения, НЕ удаляя элементы
        new_data = [val * scalar for val in self.data]
        return CSRMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование CSR матрицы через COO."""
        # Преобразуем в COO, транспонируем, затем в CSC
        coo = self._to_coo()
        coo_t = coo.transpose()
        return coo_t._to_csc()

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        if self.shape[1] != other.shape[0]:
            raise ValueError("Несовместимые размерности для умножения")
        
        rows_A, cols_A = self.shape
        rows_B, cols_B = other.shape
        
        # Преобразуем other в CSR если нужно
        if not isinstance(other, CSRMatrix):
            other_csr = CSRMatrix.from_dense(other.to_dense())
        else:
            other_csr = other
        
        # Алгоритм умножения CSR матриц
        result_data = []
        result_indices = []
        result_indptr = [0]
        
        for i in range(rows_A):
            # Словарь для накопления результатов строки
            row_result = {}
            start_A, end_A = self.indptr[i], self.indptr[i + 1]
            
            for idx_A in range(start_A, end_A):
                k = self.indices[idx_A]
                val_A = self.data[idx_A]
                
                # Добавляем вклад от строки k матрицы B
                start_B, end_B = other_csr.indptr[k], other_csr.indptr[k + 1]
                for idx_B in range(start_B, end_B):
                    j = other_csr.indices[idx_B]
                    val_B = other_csr.data[idx_B]
                    
                    row_result[j] = row_result.get(j, 0.0) + val_A * val_B
            
            # Сохраняем ненулевые элементы
            for j in sorted(row_result.keys()):
                val = row_result[j]
                if abs(val) > TOL:
                    result_data.append(val)
                    result_indices.append(j)
            
            result_indptr.append(len(result_data))
        
        return CSRMatrix(result_data, result_indices, result_indptr, (rows_A, cols_B))

    @classmethod
    def from_dense(cls, dense_matrix: List[List[float]]) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        if not dense_matrix:
            return cls([], [], [0], (0, 0))
        
        rows = len(dense_matrix)
        cols = len(dense_matrix[0])
        
        data = []
        indices = []
        indptr = [0]
        
        for i in range(rows):
            row_nnz = 0
            for j in range(cols):
                val = dense_matrix[i][j]
                if abs(val) > TOL:
                    data.append(float(val))
                    indices.append(j)
                    row_nnz += 1
            indptr.append(indptr[-1] + row_nnz)
        
        return cls(data, indices, indptr, (rows, cols))
    
    def _to_csc(self) -> 'CSCMatrix':
        """Преобразование CSRMatrix в CSCMatrix."""
        from CSC import CSCMatrix
        
        rows, cols = self.shape
        
        if self.nnz == 0:
            return CSCMatrix([], [], [0] * (cols + 1), self.shape)
        
        # Подсчитываем количество ненулевых элементов в каждом столбце
        col_counts = [0] * cols
        for j in self.indices:
            col_counts[j] += 1
        
        # Строим indptr для CSC
        indptr = [0] * (cols + 1)
        for j in range(cols):
            indptr[j + 1] = indptr[j] + col_counts[j]
        
        # Рабочие массивы для заполнения
        current_pos = indptr.copy()
        data_csc = [0.0] * self.nnz
        indices_csc = [0] * self.nnz
        
        # Заполняем CSC
        for i in range(rows):
            start, end = self.indptr[i], self.indptr[i + 1]
            for idx in range(start, end):
                j = self.indices[idx]
                pos = current_pos[j]
                data_csc[pos] = self.data[idx]
                indices_csc[pos] = i
                current_pos[j] += 1
        
        return CSCMatrix(data_csc, indices_csc, indptr, self.shape)
    
    def _to_coo(self) -> 'COOMatrix':
        """Преобразование CSRMatrix в COOMatrix."""
        from COO import COOMatrix
        
        if self.nnz == 0:
            return COOMatrix([], [], [], self.shape)
        
        data = []
        rows = []
        cols = []
        
        for i in range(self.shape[0]):
            start, end = self.indptr[i], self.indptr[i + 1]
            for idx in range(start, end):
                data.append(self.data[idx])
                rows.append(i)
                cols.append(self.indices[idx])
        
        return COOMatrix(data, rows, cols, self.shape)
