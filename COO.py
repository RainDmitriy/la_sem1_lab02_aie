from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix
from typing import Dict, Tuple, List

class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        
        if not (len(data) == len(row) == len(col)):
            raise ValueError("Длины data, row, col не равны")
        
        self.data = data
        self.row = row
        self.col = col

    def to_dense(self) -> DenseMatrix:
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]
        
        for idx in range(len(self.data)):
            dense[self.row[idx]][self.col[idx]] = self.data[idx]
        
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        sum_dict: Dict[Tuple[int, int], float] = {}
        
        # Сначала other
        for val, r, c in zip(other.data, other.row, other.col):
            sum_dict[(r, c)] = val
        
        # Затем self
        for val, r, c in zip(self.data, self.row, self.col):
            key = (r, c)
            if key in sum_dict:
                sum_dict[key] += val
            else:
                sum_dict[key] = val
        
        new_data, new_row, new_col = [], [], []
        for (r, c), val in sum_dict.items():
            if abs(val) > 1e-12:
                new_data.append(val)
                new_row.append(r)
                new_col.append(c)
        
        return COOMatrix(new_data, new_row, new_col, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        if abs(scalar) < 1e-12:
            return COOMatrix([], [], [], self.shape)
        
        new_data = [val * scalar for val in self.data]
        return COOMatrix(new_data, self.row.copy(), self.col.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        new_shape = (self.shape[1], self.shape[0])
        return COOMatrix(self.data.copy(), self.col.copy(), self.row.copy(), new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        if self.shape[1] != other.shape[0]:
            raise ValueError("Несовместимые размеры матриц")
        
        m, n = self.shape[0], other.shape[1]
        result = {}
        
        other_csr = other._to_csr()
        
        for idx in range(len(self.data)):
            row_a = self.row[idx]
            col_a = self.col[idx]
            val_a = self.data[idx]
            
            row_start = other_csr.indptr[col_a]
            row_end = other_csr.indptr[col_a + 1]
            
            for k in range(row_start, row_end):
                col_b = other_csr.indices[k]
                val_b = other_csr.data[k]
                key = (row_a, col_b)
                result[key] = result.get(key, 0.0) + val_a * val_b
        
        data, rows, cols = [], [], []
        for (i, j), val in result.items():
            if abs(val) > 1e-14:
                data.append(val)
                rows.append(i)
                cols.append(j)
        
        return COOMatrix(data, rows, cols, (m, n))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        data, row, col = [], [], []
        
        for i in range(len(dense_matrix)):
            for j in range(len(dense_matrix[0])):
                val = dense_matrix[i][j]
                if abs(val) > 1e-12:
                    data.append(val)
                    row.append(i)
                    col.append(j)
        
        return cls(data, row, col, (len(dense_matrix), len(dense_matrix[0])))

    def _to_csc(self) -> 'CSCMatrix':
        from CSC import CSCMatrix
        
        rows, cols = self.shape
        elements = list(zip(self.col, self.row, self.data))
        elements.sort()
        
        data, indices, indptr = [], [], [0] * (cols + 1)
        
        for col_idx, row_idx, val in elements:
            data.append(val)
            indices.append(row_idx)
            indptr[col_idx + 1] += 1
        
        for j in range(cols):
            indptr[j + 1] += indptr[j]
        
        return CSCMatrix(data, indices, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        from CSR import CSRMatrix
        
        rows, cols = self.shape
        elements = list(zip(self.row, self.col, self.data))
        elements.sort()
        
        data, indices, indptr = [], [], [0] * (rows + 1)
        
        for row_idx, col_idx, val in elements:
            data.append(val)
            indices.append(col_idx)
            indptr[row_idx + 1] += 1
        
        for i in range(rows):
            indptr[i + 1] += indptr[i]
        
        return CSRMatrix(data, indices, indptr, self.shape)