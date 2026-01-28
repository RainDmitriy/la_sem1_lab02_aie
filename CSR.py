from base import Matrix
from types import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix
from COO import COOMatrix
from CSC import CSCMatrix
from typing import Dict, List
import bisect


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr
        
        if len(indptr) != shape[0] + 1:
            raise ValueError(f"indptr должен иметь длину {shape[0] + 1}, получено {len(indptr)}")

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]
        
        for i in range(rows):
            row_start = self.indptr[i]
            row_end = self.indptr[i + 1]
            
            for idx in range(row_start, row_end):
                j = self.indices[idx]
                value = self.data[idx]
                dense[i][j] = value
        
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        # Используем COO для сложения
        coo_self = self._to_coo()
        if isinstance(other, CSRMatrix):
            coo_other = other._to_coo()
        else:
            coo_other = COOMatrix.from_dense(other.to_dense())
        
        result_coo = coo_self._add_impl(coo_other)
        return result_coo._to_csr()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        new_data = [val * scalar for val in self.data]
        return CSRMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Результат - в CSC формате.
        """
        # Эффективное транспонирование через COO
        coo = self._to_coo()
        transposed_coo = coo.transpose()
        return transposed_coo._to_csc()

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        if isinstance(other, CSRMatrix):
            return self._matmul_csr(other)
        elif isinstance(other, COOMatrix):
            # Конвертируем other в CSR
            other_csr = other._to_csr()
            return self._matmul_csr(other_csr)
        else:
            # Конвертируем в COO для умножения
            coo_self = self._to_coo()
            coo_other = COOMatrix.from_dense(other.to_dense())
            result_coo = coo_self._matmul_impl(coo_other)
            return result_coo._to_csr()

    def _matmul_csr(self, other: 'CSRMatrix') -> 'CSRMatrix':
        """Умножение двух CSR матриц."""
        rows_A, cols_A = self.shape
        rows_B, cols_B = other.shape
        
        if cols_A != rows_B:
            raise ValueError("Несовместимые размерности для умножения")
        
        # Подготовим B для быстрого доступа по строкам
        B_rows = []
        for i in range(rows_B):
            row_start = other.indptr[i]
            row_end = other.indptr[i + 1]
            row_dict = {}
            for idx in range(row_start, row_end):
                j = other.indices[idx]
                row_dict[j] = other.data[idx]
            B_rows.append(row_dict)
        
        # Умножаем
        result_data = []
        result_indices = []
        result_indptr = [0]
        
        for i in range(rows_A):
            # Создаем словарь для строки результата
            row_result = {}
            
            row_start = self.indptr[i]
            row_end = self.indptr[i + 1]
            
            for idx_A in range(row_start, row_end):
                k = self.indices[idx_A]
                val_A = self.data[idx_A]
                
                # Умножаем на строку k матрицы B
                for j, val_B in B_rows[k].items():
                    product = val_A * val_B
                    if j in row_result:
                        row_result[j] += product
                    else:
                        row_result[j] = product
            
            # Добавляем ненулевые элементы в результат
            sorted_cols = sorted(row_result.keys())
            for j in sorted_cols:
                val = row_result[j]
                if abs(val) > 1e-12:
                    result_data.append(val)
                    result_indices.append(j)
            
            result_indptr.append(len(result_data))
        
        return CSRMatrix(result_data, result_indices, result_indptr, (rows_A, cols_B))

    def get(self, i: int, j: int) -> float:
        """Получить элемент (i, j)."""
        row_start = self.indptr[i]
        row_end = self.indptr[i + 1]
        
        # Бинарный поиск в отсортированных индексах строки
        left, right = row_start, row_end - 1
        while left <= right:
            mid = (left + right) // 2
            if self.indices[mid] == j:
                return self.data[mid]
            elif self.indices[mid] < j:
                left = mid + 1
            else:
                right = mid - 1
        
        return 0.0

    def get_row_as_dict(self, i: int) -> Dict[int, float]:
        """Получить строку i в виде словаря."""
        row_dict = {}
        row_start = self.indptr[i]
        row_end = self.indptr[i + 1]
        
        for idx in range(row_start, row_end):
            j = self.indices[idx]
            row_dict[j] = self.data[idx]
        
        return row_dict

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        data = []
        indices = []
        indptr = [0]
        
        for i, row_vals in enumerate(dense_matrix):
            row_nnz = 0
            for j, val in enumerate(row_vals):
                if abs(val) > 1e-12:
                    data.append(val)
                    indices.append(j)
                    row_nnz += 1
            indptr.append(indptr[-1] + row_nnz)
        
        shape = (len(dense_matrix), len(dense_matrix[0]) if dense_matrix else 0)
        return cls(data, indices, indptr, shape)

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSRMatrix в CSCMatrix.
        """
        # Используем COO как промежуточный формат
        coo = self._to_coo()
        return coo._to_csc()
    
    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSRMatrix в COOMatrix.
        """
        data = []
        row = []
        col = []
        
        rows = self.shape[0]
        for i in range(rows):
            row_start = self.indptr[i]
            row_end = self.indptr[i + 1]
            
            for idx in range(row_start, row_end):
                data.append(self.data[idx])
                row.append(i)
                col.append(self.indices[idx])
        
        return COOMatrix(data, row, col, self.shape)