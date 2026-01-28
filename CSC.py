from base import Matrix
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix
from COO import COOMatrix
from CSR import CSRMatrix
from typing import Dict, List
import bisect


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr
        
        if len(indptr) != shape[1] + 1:
            raise ValueError(f"indptr должен иметь длину {shape[1] + 1}, получено {len(indptr)}")

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]
        
        for j in range(cols):
            col_start = self.indptr[j]
            col_end = self.indptr[j + 1]
            
            for idx in range(col_start, col_end):
                i = self.indices[idx]
                value = self.data[idx]
                dense[i][j] = value
        
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        # Используем COO для сложения
        coo_self = self._to_coo()
        if isinstance(other, CSCMatrix):
            coo_other = other._to_coo()
        else:
            coo_other = COOMatrix.from_dense(other.to_dense())
        
        result_coo = coo_self._add_impl(coo_other)
        return result_coo._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        new_data = [val * scalar for val in self.data]
        return CSCMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Результат - в CSR формате.
        """
        # Эффективное транспонирование через COO
        coo = self._to_coo()
        transposed_coo = coo.transpose()
        return transposed_coo._to_csr()

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        # Конвертируем в CSR для умножения
        csr_self = self._to_csr()
        if isinstance(other, CSCMatrix):
            csr_other = other._to_csr()
            result_csr = csr_self._matmul_impl(csr_other)
            return result_csr._to_csc()
        else:
            result_csr = csr_self._matmul_impl(other)
            return result_csr._to_csc()

    def get(self, i: int, j: int) -> float:
        """Получить элемент (i, j)."""
        col_start = self.indptr[j]
        col_end = self.indptr[j + 1]
        
        # Бинарный поиск в отсортированных индексах столбца
        left, right = col_start, col_end - 1
        while left <= right:
            mid = (left + right) // 2
            if self.indices[mid] == i:
                return self.data[mid]
            elif self.indices[mid] < i:
                left = mid + 1
            else:
                right = mid - 1
        
        return 0.0

    def get_column_as_dict(self, j: int) -> Dict[int, float]:
        """Получить столбец j в виде словаря."""
        col_dict = {}
        col_start = self.indptr[j]
        col_end = self.indptr[j + 1]
        
        for idx in range(col_start, col_end):
            i = self.indices[idx]
            col_dict[i] = self.data[idx]
        
        return col_dict

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        if not dense_matrix:
            return cls([], [], [0], (0, 0))
        
        rows = len(dense_matrix)
        cols = len(dense_matrix[0])
        
        # Собираем данные по столбцам
        col_data = [[] for _ in range(cols)]
        col_indices = [[] for _ in range(cols)]
        
        for i in range(rows):
            for j in range(cols):
                val = dense_matrix[i][j]
                if abs(val) > 1e-12:
                    col_data[j].append(val)
                    col_indices[j].append(i)
        
        # Преобразуем в плоские массивы
        data = []
        indices = []
        indptr = [0]
        
        for j in range(cols):
            data.extend(col_data[j])
            indices.extend(col_indices[j])
            indptr.append(len(data))
        
        return cls(data, indices, indptr, (rows, cols))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        # Используем COO как промежуточный формат
        coo = self._to_coo()
        return coo._to_csr()

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        data = []
        row = []
        col = []
        
        cols = self.shape[1]
        for j in range(cols):
            col_start = self.indptr[j]
            col_end = self.indptr[j + 1]
            
            for idx in range(col_start, col_end):
                data.append(self.data[idx])
                row.append(self.indices[idx])
                col.append(j)
        
        return COOMatrix(data, row, col, self.shape)