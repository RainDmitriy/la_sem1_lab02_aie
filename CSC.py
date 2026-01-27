from base import Matrix
from mytypes import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix
from typing import List, Dict
from collections import defaultdict

class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        
        n, m = shape
        if len(indptr) != m + 1:
            raise ValueError(f"indptr должен иметь длину {m + 1}")
        if indptr[0] != 0:
            raise ValueError("indptr[0] должен быть 0")
        if indptr[-1] != len(data):
            raise ValueError(f"indptr[-1] должен быть равен {len(data)}")
        if len(data) != len(indices):
            raise ValueError("Длины data и indices должны совпадать")
        
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.nnz = len(data)

    def to_dense(self) -> DenseMatrix:
        n, m = self.shape
        dense = [[0.0] * m for _ in range(n)]
        
        for j in range(m):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for idx in range(start, end):
                i = self.indices[idx]
                dense[i][j] = self.data[idx]
        
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        self_coo = self._to_coo()
        other_coo = self._convert_to_coo(other)
        
        result_coo = self_coo._add_impl(other_coo)
        return result_coo._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        if abs(scalar) < 1e-12:
            _, m = self.shape
            return CSCMatrix([], [], [0] * (m + 1), self.shape)
        
        new_data = [val * scalar for val in self.data]
        return CSCMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        from CSR import CSRMatrix
        
        n, m = self.shape
        new_shape = (m, n)

        row_counts = [0] * m
        for i in self.indices:
            row_counts[i] += 1

        new_indptr = [0] * (m + 1)
        for i in range(m):
            new_indptr[i + 1] = new_indptr[i] + row_counts[i]

        new_data = [0.0] * self.nnz
        new_indices = [0] * self.nnz
        positions = new_indptr.copy()
        
        for j in range(n):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for idx in range(start, end):
                i = self.indices[idx]
                pos = positions[i]
                new_data[pos] = self.data[idx]
                new_indices[pos] = j
                positions[i] += 1
        
        return CSRMatrix(new_data, new_indices, new_indptr, new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        self_coo = self._to_coo()
        other_coo = self._convert_to_coo(other)
        
        result_coo = self_coo._matmul_impl(other_coo)
        return result_coo._to_csc()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        from COO import COOMatrix
        coo = COOMatrix.from_dense(dense_matrix)
        return coo._to_csc()

    def _to_csr(self) -> 'CSRMatrix':
        from CSR import CSRMatrix
        coo = self._to_coo()
        return coo._to_csr()

    def _to_coo(self) -> 'COOMatrix':
        from COO import COOMatrix
        
        n, m = self.shape
        data, rows, cols = [], [], []
        
        for j in range(m):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for idx in range(start, end):
                data.append(self.data[idx])
                rows.append(self.indices[idx])
                cols.append(j)
        
        return COOMatrix(data, rows, cols, self.shape)

    def _convert_to_coo(self, other: 'Matrix') -> 'COOMatrix':
        """Вспомогательный метод для преобразования к COO."""
        if hasattr(other, '_to_coo'):
            return other._to_coo()
        try:
            other_dense = other.to_dense()
            from COO import COOMatrix
            return COOMatrix.from_dense(other_dense)
        except:
            raise ValueError("Невозможно преобразовать матрицу к COO")