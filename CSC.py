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

        if self.nnz == 0:
            return CSRMatrix([], [], [0] * (m + 1), new_shape)

        row_counts = [0] * m
        for col in range(m):
            start = self.indptr[col]
            end = self.indptr[col + 1]
            row_counts[col] = end - start

        new_indptr = [0] * (m + 1)
        for i in range(m):
            new_indptr[i + 1] = new_indptr[i] + row_counts[i]

        new_data = [0.0] * self.nnz
        new_indices = [0] * self.nnz

        current_pos = new_indptr.copy()

        for col in range(m):
            start = self.indptr[col]
            end = self.indptr[col + 1]
            
            for idx in range(start, end):
                row = self.indices[idx]
                value = self.data[idx]

                pos = current_pos[col]
                new_data[pos] = value
                new_indices[pos] = row
                current_pos[col] += 1
        
        return CSRMatrix(new_data, new_indices, new_indptr, new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        if self.shape[1] != other.shape[0]:
            raise ValueError("Размеры матриц несовместимы для умножения")

        if not isinstance(other, CSCMatrix):
            if hasattr(other, '_to_csc'):
                other = other._to_csc()
            else:
                other_coo = self._convert_to_coo(other)
                other = other_coo._to_csc()
        
        rows_A, cols_A = self.shape
        rows_B, cols_B = other.shape

        B_by_col = {}
        for j in range(cols_B):
            start = other.indptr[j]
            end = other.indptr[j + 1]
            col_elems = []
            for idx in range(start, end):
                i = other.indices[idx]
                col_elems.append((i, other.data[idx]))
            if col_elems:
                B_by_col[j] = col_elems
        
        result_data = []
        result_indices = []
        result_indptr = [0]

        for j in range(cols_B):
            if j not in B_by_col:
                result_indptr.append(len(result_data))
                continue

            col_result = [0.0] * rows_A

            for row_B, val_B in B_by_col[j]:
                start_A = self.indptr[row_B]
                end_A = self.indptr[row_B + 1]
                for idx_A in range(start_A, end_A):
                    row_A = self.indices[idx_A]
                    val_A = self.data[idx_A]
                    col_result[row_A] += val_A * val_B

            for i in range(rows_A):
                if abs(col_result[i]) > 1e-12:
                    result_data.append(col_result[i])
                    result_indices.append(i)
            
            result_indptr.append(len(result_data))
        
        return CSCMatrix(result_data, result_indices, result_indptr, (rows_A, cols_B))

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
    
    def _convert_to_csc(self, other: 'Matrix') -> 'CSCMatrix':
        """Вспомогательный метод для преобразования к CSC."""
        if isinstance(other, CSCMatrix):
            return other
        
        if hasattr(other, '_to_csc'):
            return other._to_csc()
        try:
            coo = other._to_coo()
            return coo._to_csc()
        except:
            raise ValueError("Невозможно преобразовать матрицу к CSC")