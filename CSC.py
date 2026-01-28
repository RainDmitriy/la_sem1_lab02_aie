from base import Matrix
from typing import List, Tuple

TOL = 1e-12


class CSCMatrix(Matrix):
    def __init__(self, data: List[float], indices: List[int], indptr: List[int], shape: Tuple[int, int]):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.nnz = len(data)
        
        # Проверка целостности
        if len(indptr) != shape[1] + 1:
            raise ValueError(f"Длина indptr должна быть {shape[1] + 1}, получено {len(indptr)}")
        if indices and max(indices) >= shape[0]:
            raise ValueError(f"Индекс строки {max(indices)} превышает размер {shape[0]}")

    def to_dense(self) -> List[List[float]]:
        """Преобразует CSC в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]
        
        for j in range(cols):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for idx in range(start, end):
                row = self.indices[idx]
                dense[row][j] = self.data[idx]
        
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        if self.shape != other.shape:
            raise ValueError("Размерности матриц не совпадают")
        
        # Если other тоже CSC
        if isinstance(other, CSCMatrix):
            rows, cols = self.shape
            result_data = []
            result_indices = []
            result_indptr = [0]
            
            for j in range(cols):
                self_start = self.indptr[j]
                self_end = self.indptr[j + 1]
                other_start = other.indptr[j]
                other_end = other.indptr[j + 1]
                
                idx1, idx2 = self_start, other_start
                
                while idx1 < self_end and idx2 < other_end:
                    row1 = self.indices[idx1]
                    row2 = other.indices[idx2]
                    
                    if row1 < row2:
                        result_data.append(self.data[idx1])
                        result_indices.append(row1)
                        idx1 += 1
                    elif row1 > row2:
                        result_data.append(other.data[idx2])
                        result_indices.append(row2)
                        idx2 += 1
                    else:
                        val = self.data[idx1] + other.data[idx2]
                        if abs(val) > TOL:
                            result_data.append(val)
                            result_indices.append(row1)
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
            
            return CSCMatrix(result_data, result_indices, result_indptr, self.shape)
        else:
            # Иначе преобразуем в плотные
            dense_self = self.to_dense()
            dense_other = other.to_dense()
            rows, cols = self.shape
            
            data = []
            indices = []
            indptr = [0]
            
            for j in range(cols):
                col_nnz = 0
                for i in range(rows):
                    val = dense_self[i][j] + dense_other[i][j]
                    if abs(val) > TOL:
                        data.append(val)
                        indices.append(i)
                        col_nnz += 1
                indptr.append(indptr[-1] + col_nnz)
            
            return CSCMatrix(data, indices, indptr, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        if abs(scalar) < TOL:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)
        
        # Умножаем все значения, удаляя элементы близкие к нулю
        new_data = []
        new_indices = []
        new_indptr = [0]
        
        for j in range(self.shape[1]):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            col_nnz = 0
            
            for idx in range(start, end):
                val = self.data[idx] * scalar
                if abs(val) > TOL:
                    new_data.append(val)
                    new_indices.append(self.indices[idx])
                    col_nnz += 1
            
            new_indptr.append(new_indptr[-1] + col_nnz)
        
        return CSCMatrix(new_data, new_indices, new_indptr, self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование CSC матрицы через COO."""
        # Преобразуем в COO, транспонируем, затем в CSR
        coo = self._to_coo()
        coo_t = coo.transpose()
        return coo_t._to_csr()

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        if self.shape[1] != other.shape[0]:
            raise ValueError("Несовместимые размерности для умножения")
        
        # Преобразуем в CSR для умножения
        csr_self = self._to_csr()
        result_csr = csr_self._matmul_impl(other)
        return result_csr._to_csc()

    @classmethod
    def from_dense(cls, dense_matrix: List[List[float]]) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        if not dense_matrix:
            return cls([], [], [0], (0, 0))
        
        rows = len(dense_matrix)
        cols = len(dense_matrix[0])
        
        data = []
        indices = []
        indptr = [0]
        
        for j in range(cols):
            col_nnz = 0
            for i in range(rows):
                val = dense_matrix[i][j]
                if abs(val) > TOL:
                    data.append(float(val))
                    indices.append(i)
                    col_nnz += 1
            indptr.append(indptr[-1] + col_nnz)
        
        return cls(data, indices, indptr, (rows, cols))

    def _to_csr(self) -> 'CSRMatrix':
        """Преобразование CSCMatrix в CSRMatrix."""
        from CSR import CSRMatrix
        
        rows, cols = self.shape
        
        if self.nnz == 0:
            return CSRMatrix([], [], [0] * (rows + 1), self.shape)
        
        # Подсчитываем количество ненулевых элементов в каждой строке
        row_counts = [0] * rows
        for i in self.indices:
            row_counts[i] += 1
        
        # Строим indptr для CSR
        indptr = [0] * (rows + 1)
        for i in range(rows):
            indptr[i + 1] = indptr[i] + row_counts[i]
        
        # Рабочие массивы для заполнения
        current_pos = indptr.copy()
        data_csr = [0.0] * self.nnz
        indices_csr = [0] * self.nnz
        
        # Заполняем CSR
        for j in range(cols):
            start, end = self.indptr[j], self.indptr[j + 1]
            for idx in range(start, end):
                i = self.indices[idx]
                pos = current_pos[i]
                data_csr[pos] = self.data[idx]
                indices_csr[pos] = j
                current_pos[i] += 1
        
        return CSRMatrix(data_csr, indices_csr, indptr, self.shape)

    def _to_coo(self) -> 'COOMatrix':
        """Преобразование CSCMatrix в COOMatrix."""
        from COO import COOMatrix
        
        if self.nnz == 0:
            return COOMatrix([], [], [], self.shape)
        
        data = []
        rows = []
        cols = []
        
        for j in range(self.shape[1]):
            start, end = self.indptr[j], self.indptr[j + 1]
            for idx in range(start, end):
                data.append(self.data[idx])
                rows.append(self.indices[idx])
                cols.append(j)
        
        return COOMatrix(data, rows, cols, self.shape)
