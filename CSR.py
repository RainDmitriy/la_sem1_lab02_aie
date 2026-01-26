from base import Matrix
from types import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix
from collections import defaultdict
from typing import List, Tuple

class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        n, m = shape
        if len(indptr) != n + 1:
            raise ValueError(f"indptr должен иметь длину n+1 = {n+1}, получено {len(indptr)}")

        for i in range(n):
            if indptr[i] > indptr[i + 1]:
                raise ValueError(f"indptr должен быть неубывающим: indptr[{i}] = {indptr[i]} > indptr[{i+1}] = {indptr[i+1]}")

        if indptr[0] != 0:
            raise ValueError(f"indptr[0] должен быть 0, получено {indptr[0]}")
        if indptr[-1] != len(data):
            raise ValueError(f"indptr[-1] должен быть равен len(data) = {len(data)}, получено {indptr[-1]}")

        if len(data) != len(indices):
            raise ValueError(f"data и indices должны быть одинаковой длины: data={len(data)}, indices={len(indices)}")

        for col_idx in indices:
            if not (0 <= col_idx < m):
                raise ValueError(f"Индекс столбца {col_idx} вне диапазона [0, {m-1}]")
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.nnz = len(data)

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        n, m = self.shape
        if n * m > 10000:
            raise MemoryError(f"Матрица {n}x{m} слишком большая для dense")
        
        mat = [[0.0] * m for _ in range(n)]
        for i in range(n):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for idx in range(start, end):
                j = self.indices[idx]
                mat[i][j] = self.data[idx]
        return mat

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        if not isinstance(other, CSRMatrix):
            if hasattr(other, '_to_csr'):
                other = other._to_csr()
            else:
                if self.shape[0] * self.shape[1] <= 10000:
                    from COO import COOMatrix
                    other_coo = COOMatrix.from_dense(other.to_dense())
                    other = other_coo._to_csr()
                else:
                    raise ValueError("Нельзя складывать большие матрицы через dense")
        
        n, m = self.shape
        result_data, result_indices, result_indptr = [], [], [0]
        
        for i in range(n):
            p1, p2 = self.indptr[i], other.indptr[i]
            end1, end2 = self.indptr[i + 1], other.indptr[i + 1]
            
            while p1 < end1 and p2 < end2:
                j1, j2 = self.indices[p1], other.indices[p2]
                
                if j1 < j2:
                    result_data.append(self.data[p1])
                    result_indices.append(j1)
                    p1 += 1
                elif j1 > j2:
                    result_data.append(other.data[p2])
                    result_indices.append(j2)
                    p2 += 1
                else:
                    val = self.data[p1] + other.data[p2]
                    if abs(val) > 1e-12:
                        result_data.append(val)
                        result_indices.append(j1)
                    p1 += 1
                    p2 += 1
            
            while p1 < end1:
                result_data.append(self.data[p1])
                result_indices.append(self.indices[p1])
                p1 += 1
            
            while p2 < end2:
                result_data.append(other.data[p2])
                result_indices.append(other.indices[p2])
                p2 += 1
            
            result_indptr.append(len(result_data))
        
        return CSRMatrix(result_data, result_indices, result_indptr, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        new_data = [x * scalar for x in self.data]
        return CSRMatrix(new_data, self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Hint:
        Результат - в CSC формате (с теми же данными, но с интерпретацией столбцов как строк).
        """
        from CSC import CSCMatrix
        
        n, m = self.shape
        nnz = self.nnz
        
        col_counts = [0] * m
        for col in self.indices:
            col_counts[col] += 1
        
        indptr_csc = [0] * (m + 1)
        for j in range(m):
            indptr_csc[j + 1] = indptr_csc[j] + col_counts[j]
        
        current_pos = indptr_csc.copy()
        data_csc = [0.0] * nnz
        indices_csc = [0] * nnz
        
        for i in range(n):
            for pos in range(self.indptr[i], self.indptr[i + 1]):
                j = self.indices[pos]
                csc_pos = current_pos[j]
                data_csc[csc_pos] = self.data[pos]
                indices_csc[csc_pos] = i
                current_pos[j] += 1
        
        return CSCMatrix(data_csc, indices_csc, indptr_csc, (m, n))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        if not isinstance(other, CSRMatrix):
            if hasattr(other, '_to_csr'):
                other_csr = other._to_csr()
            else:
                if self.shape[0] * self.shape[1] <= 10000 and \
                   other.shape[0] * other.shape[1] <= 10000:
                    from COO import COOMatrix
                    other_coo = COOMatrix.from_dense(other.to_dense())
                    other_csr = other_coo._to_csr()
                else:
                    raise ValueError("Нельзя умножать большие матрицы через dense")
        else:
            other_csr = other
        
        n, k = self.shape
        _, m = other_csr.shape

        other_csc = other_csr.transpose()
        
        result_data, result_indices, result_indptr = [], [], [0]
        
        for i in range(n):
            row_result = defaultdict(float)
            
            for p in range(self.indptr[i], self.indptr[i + 1]):
                col_a = self.indices[p]
                val_a = self.data[p]
                
                start_b = other_csc.indptr[col_a]
                end_b = other_csc.indptr[col_a + 1]
                
                for q in range(start_b, end_b):
                    row_b = other_csc.indices[q]
                    val_b = other_csc.data[q]
                    row_result[row_b] += val_a * val_b
            
            sorted_cols = sorted(row_result.keys())
            for col in sorted_cols:
                val = row_result[col]
                if abs(val) > 1e-12:
                    result_data.append(val)
                    result_indices.append(col)
            
            result_indptr.append(len(result_data))
        
        return CSRMatrix(result_data, result_indices, result_indptr, (n, m))

    """Методы для LU"""
    def get_row(self, i: int) -> List[Tuple[int, float]]:
        start = self.indptr[i]
        end = self.indptr[i + 1]
        return [(self.indices[idx], self.data[idx]) for idx in range(start, end)]

    def set_row(self, i: int, elements: List[Tuple[int, float]]):
        n, m = self.shape
        
        all_rows = []
        for row_idx in range(n):
            if row_idx == i:
                all_rows.append(sorted(elements, key=lambda x: x[0]))
            else:
                all_rows.append(self.get_row(row_idx))
        
        new_data, new_indices, new_indptr = [], [], [0]
        
        for row in all_rows:
            for col, val in row:
                if abs(val) > 1e-12:
                    new_data.append(val)
                    new_indices.append(col)
            new_indptr.append(len(new_data))
        
        self.data = new_data
        self.indices = new_indices
        self.indptr = new_indptr
        self.nnz = len(new_data)

    def swap_rows(self, i: int, j: int):
        if i == j:
            return
        
        row_i = self.get_row(i)
        row_j = self.get_row(j)
        
        self.set_row(i, row_j)
        self.set_row(j, row_i)

    def get_element(self, i: int, j: int) -> float:
        start = self.indptr[i]
        end = self.indptr[i + 1]
        
        left, right = start, end - 1
        while left <= right:
            mid = (left + right) // 2
            col = self.indices[mid]
            if col == j:
                return self.data[mid]
            elif col < j:
                left = mid + 1
            else:
                right = mid - 1
        return 0.0

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        from COO import COOMatrix
        return COOMatrix.from_dense(dense_matrix)._to_csr()

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSRMatrix в CSCMatrix.
        """
        return self.transpose()

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSRMatrix в COOMatrix.
        """
        from COO import COOMatrix
        
        n, m = self.shape
        data, row_indices, col_indices = [], [], []
        
        for i in range(n):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for idx in range(start, end):
                data.append(self.data[idx])
                row_indices.append(i)
                col_indices.append(self.indices[idx])
        
        return COOMatrix(data, row_indices, col_indices, self.shape)
    