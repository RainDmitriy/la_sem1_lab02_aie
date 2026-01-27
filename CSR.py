from base import Matrix, DenseMatrix, Shape
from typing import List
from COO import COOMatrix
from CSC import CSCMatrix

CSRData = List[float]
CSRIndices = List[int]
CSRIndptr = List[int]


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
            for j in range(start, end):
                col = self.indices[j]
                dense[i][col] = self.data[j]
        
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        # Преобразуем в COO для сложения
        coo_self = self._to_coo()
        return coo_self._add_impl(other)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        if scalar == 0:
            return CSRMatrix([], [], [0] * (self.rows + 1), self.shape)
        new_data = [val * scalar for val in self.data]
        return CSRMatrix(new_data, self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        # Транспонирование CSR -> CSC
        return self._to_csc()

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        from CSC import CSCMatrix
        
        if isinstance(other, CSCMatrix):
            # CSR * CSC
            result_rows = self.rows
            result_cols = other.cols
            result_data = []
            result_indices = []
            result_indptr = [0]
            
            for i in range(self.rows):
                row_start = self.indptr[i]
                row_end = self.indptr[i + 1]
                row_nnz = 0
                
                # Вектор для аккумуляции результата строки
                row_result = [0.0] * result_cols
                
                # Умножаем строку i на столбцы other
                for k in range(row_start, row_end):
                    col_in_self = self.indices[k]
                    val_in_self = self.data[k]
                    
                    # Получаем столбец col_in_self из other
                    col_start = other.indptr[col_in_self]
                    col_end = other.indptr[col_in_self + 1]
                    
                    for l in range(col_start, col_end):
                        j = other.indices[l]
                        row_result[j] += val_in_self * other.data[l]
                
                # Формируем CSR строку результата
                for j in range(result_cols):
                    if row_result[j] != 0:
                        result_data.append(row_result[j])
                        result_indices.append(j)
                        row_nnz += 1
                
                result_indptr.append(result_indptr[-1] + row_nnz)
            
            return CSRMatrix(result_data, result_indices, result_indptr, (result_rows, result_cols))
        else:
            # Для других форматов преобразуем в плотный
            dense_other = other.to_dense()
            result_rows = self.rows
            result_cols = other.cols
            result = [[0.0] * result_cols for _ in range(result_rows)]
            
            for i in range(self.rows):
                row_start = self.indptr[i]
                row_end = self.indptr[i + 1]
                
                for k in range(row_start, row_end):
                    col_in_self = self.indices[k]
                    val_in_self = self.data[k]
                    
                    for j in range(result_cols):
                        result[i][j] += val_in_self * dense_other[col_in_self][j]
            
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
                if val != 0:
                    data.append(val)
                    indices.append(j)
                    nnz_in_row += 1
            indptr.append(indptr[-1] + nnz_in_row)
        
        return cls(data, indices, indptr, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
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
        
        return CSCMatrix(csc_data, csc_indices, csc_indptr, (rows, cols))

    def _to_coo(self) -> 'COOMatrix':
        data = self.data[:]
        rows = []
        cols = self.indices[:]
        
        for i in range(self.rows):
            for k in range(self.indptr[i], self.indptr[i + 1]):
                rows.append(i)
        
        return COOMatrix(data, rows, cols, self.shape)
