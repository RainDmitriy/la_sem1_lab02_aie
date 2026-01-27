from base import Matrix
from mytypes import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix
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
        """Сложение через преобразование в COO."""
        self_coo = self._to_coo()

        if isinstance(other, CSRMatrix):
            other_coo = other._to_coo()
        elif hasattr(other, '_to_coo'):
            other_coo = other._to_coo()
        else:
            if self.shape[0] * self.shape[1] <= 10000:
                from COO import COOMatrix
                other_coo = COOMatrix.from_dense(other.to_dense())
            else:
                raise ValueError("Нельзя складывать большие матрицы через dense")

        result_coo = self_coo._add_impl(other_coo)

        return result_coo._to_csr()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        new_data = [x * scalar for x in self.data]
        return CSRMatrix(new_data, self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование через COO."""
        from CSC import CSCMatrix

        coo = self._to_coo()

        transposed_coo = COOMatrix(
            coo.data[:],
            coo.col[:],
            coo.row[:],
            (self.shape[1], self.shape[0])
        )
        return transposed_coo._to_csc()

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение через преобразование в COO."""

        if isinstance(other, CSRMatrix):
            other_coo = other._to_coo()
        elif hasattr(other, '_to_coo'):
            other_coo = other._to_coo()
        else:
            if self.shape[0] * self.shape[1] <= 10000 and \
            other.shape[0] * other.shape[1] <= 10000:
                from COO import COOMatrix
                other_coo = COOMatrix.from_dense(other.to_dense())
            else:
                raise ValueError("Нельзя умножать большие матрицы через dense")
        
        result_coo = self_coo._matmul_impl(other_coo)

        return result_coo._to_csr()

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
    