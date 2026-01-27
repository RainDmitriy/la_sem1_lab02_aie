from base import Matrix
from mytypes import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix
from typing import List, Dict, Tuple
from collections import defaultdict

class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        
        n, m = shape
        if len(indptr) != n + 1:
            raise ValueError(f"indptr должен иметь длину {n + 1}")
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
        
        for i in range(n):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for idx in range(start, end):
                j = self.indices[idx]
                dense[i][j] = self.data[idx]
        
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        if not isinstance(other, CSRMatrix):
            other_csr = self._convert_to_csr(other)
            other = other_csr
        
        n, m = self.shape
        result_data, result_indices, result_indptr = [], [], [0]
        
        for i in range(n):
            p1, p2 = self.indptr[i], other.indptr[i]
            end1, end2 = self.indptr[i + 1], other.indptr[i + 1]
            
            merged: Dict[int, float] = {}

            while p1 < end1:
                j = self.indices[p1]
                merged[j] = self.data[p1]
                p1 += 1

            while p2 < end2:
                j = other.indices[p2]
                if j in merged:
                    merged[j] += other.data[p2]
                else:
                    merged[j] = other.data[p2]
                p2 += 1

            sorted_items = sorted((col, val) for col, val in merged.items() if abs(val) > 1e-12)
            for col, val in sorted_items:
                result_data.append(val)
                result_indices.append(col)
            
            result_indptr.append(len(result_data))
        
        return CSRMatrix(result_data, result_indices, result_indptr, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        if abs(scalar) < 1e-12:
            n, _ = self.shape
            return CSRMatrix([], [], [0] * (n + 1), self.shape)
        
        new_data = [val * scalar for val in self.data]
        return CSRMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        from CSC import CSCMatrix
        
        n, m = self.shape
        new_shape = (m, n)

        col_counts = [0] * m
        for j in self.indices:
            col_counts[j] += 1

        new_indptr = [0] * (m + 1)
        for j in range(m):
            new_indptr[j + 1] = new_indptr[j] + col_counts[j]

        new_data = [0.0] * self.nnz
        new_indices = [0] * self.nnz
        positions = new_indptr.copy()
        
        for i in range(n):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for idx in range(start, end):
                j = self.indices[idx]
                pos = positions[j]
                new_data[pos] = self.data[idx]
                new_indices[pos] = i
                positions[j] += 1
        
        return CSCMatrix(new_data, new_indices, new_indptr, new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        if not isinstance(other, CSRMatrix):
            other_csr = self._convert_to_csr(other)
            other = other_csr
        
        rows_A, cols_A = self.shape
        rows_B, cols_B = other.shape
        
        if cols_A != rows_B:
            raise ValueError("Размеры матриц несовместимы для умножения")
        
        result_data, result_indices, result_indptr = [], [], [0]

        B_by_row = []
        for i in range(rows_B):
            start = other.indptr[i]
            end = other.indptr[i + 1]
            row_dict = {}
            for idx in range(start, end):
                j = other.indices[idx]
                row_dict[j] = other.data[idx]
            B_by_row.append(row_dict)
        
        for i in range(rows_A):
            row_start = self.indptr[i]
            row_end = self.indptr[i + 1]

            row_result: Dict[int, float] = {}

            for idx_A in range(row_start, row_end):
                k = self.indices[idx_A]
                val_A = self.data[idx_A]

                if k < len(B_by_row):
                    for j, val_B in B_by_row[k].items():
                        if j not in row_result:
                            row_result[j] = 0.0
                        row_result[j] += val_A * val_B

            sorted_cols = sorted(col for col in row_result if abs(row_result[col]) > 1e-12)
            for col in sorted_cols:
                result_data.append(row_result[col])
                result_indices.append(col)
            
            result_indptr.append(len(result_data))
        
        return CSRMatrix(result_data, result_indices, result_indptr, (rows_A, cols_B))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        from COO import COOMatrix
        coo = COOMatrix.from_dense(dense_matrix)
        return coo._to_csr()

    def _to_csc(self) -> 'CSCMatrix':
        from CSC import CSCMatrix
        coo = self._to_coo()
        return coo._to_csc()

    def _to_coo(self) -> 'COOMatrix':
        from COO import COOMatrix
        
        n, m = self.shape
        data, rows, cols = [], [], []
        
        for i in range(n):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for idx in range(start, end):
                data.append(self.data[idx])
                rows.append(i)
                cols.append(self.indices[idx])
        
        return COOMatrix(data, rows, cols, self.shape)

    def _convert_to_csr(self, other: 'Matrix') -> 'CSRMatrix':
        """Вспомогательный метод для преобразования к CSR."""
        if isinstance(other, CSRMatrix):
            return other
        
        if hasattr(other, '_to_csr'):
            return other._to_csr()

        try:
            coo = other._to_coo()
            return coo._to_csr()
        except:
            raise ValueError("Невозможно преобразовать матрицу к CSR")