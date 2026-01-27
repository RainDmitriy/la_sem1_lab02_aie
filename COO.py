from base import Matrix
from mytypes import COOData, COORows, COOCols, Shape, DenseMatrix
from typing import Dict, Tuple, List
from collections import defaultdict

class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        
        if not (len(data) == len(row) == len(col)):
            raise ValueError("Длины data, row, col не равны")
        
        n, m = shape
        for r in row:
            if not (0 <= r < n):
                raise ValueError(f"Индекс строки {r} вне диапазона")
        for c in col:
            if not (0 <= c < m):
                raise ValueError(f"Индекс столбца {c} вне диапазона")
        
        self.data = data
        self.row = row
        self.col = col
        self.nnz = len(data)

    def to_dense(self) -> DenseMatrix:
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]
        
        for idx in range(self.nnz):
            i = self.row[idx]
            j = self.col[idx]
            dense[i][j] = self.data[idx]
        
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        other_coo = self._convert_to_coo(other)

        combined: Dict[Tuple[int, int], float] = {}

        for idx in range(self.nnz):
            key = (self.row[idx], self.col[idx])
            combined[key] = self.data[idx]

        for idx in range(other_coo.nnz):
            key = (other_coo.row[idx], other_coo.col[idx])
            if key in combined:
                combined[key] += other_coo.data[idx]
            else:
                combined[key] = other_coo.data[idx]

        new_data, new_rows, new_cols = [], [], []
        for (i, j), val in combined.items():
            if abs(val) > 1e-12:
                new_data.append(val)
                new_rows.append(i)
                new_cols.append(j)
        
        return COOMatrix(new_data, new_rows, new_cols, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        if abs(scalar) < 1e-12:
            return COOMatrix([], [], [], self.shape)
        
        new_data = [val * scalar for val in self.data]
        return COOMatrix(new_data, self.row.copy(), self.col.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        new_shape = (self.shape[1], self.shape[0])
        return COOMatrix(self.data.copy(), self.col.copy(), self.row.copy(), new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        other_coo = self._convert_to_coo(other)
        
        rows_A, cols_A = self.shape
        rows_B, cols_B = other_coo.shape
        
        if cols_A != rows_B:
            raise ValueError("Размеры матриц несовместимы для умножения")

        A_by_row = defaultdict(list)
        for idx in range(self.nnz):
            A_by_row[self.row[idx]].append((self.col[idx], self.data[idx]))
        
        B_by_col = defaultdict(list)
        for idx in range(other_coo.nnz):
            B_by_col[other_coo.col[idx]].append((other_coo.row[idx], other_coo.data[idx]))

        result = defaultdict(float)
        for i, A_elems in A_by_row.items():
            for k, val_A in A_elems:
                if k in B_by_col:
                    for j, val_B in B_by_col[k]:
                        result[(i, j)] += val_A * val_B

        new_data, new_rows, new_cols = [], [], []
        for (i, j), val in result.items():
            if abs(val) > 1e-12:
                new_data.append(val)
                new_rows.append(i)
                new_cols.append(j)
        
        return COOMatrix(new_data, new_rows, new_cols, (rows_A, cols_B))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        
        data, row_indices, col_indices = [], [], []
        
        for i in range(rows):
            for j in range(cols):
                val = dense_matrix[i][j]
                if abs(val) > 1e-12:
                    data.append(val)
                    row_indices.append(i)
                    col_indices.append(j)
        
        return cls(data, row_indices, col_indices, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        from CSC import CSCMatrix
        
        if self.nnz == 0:
            _, m = self.shape
            return CSCMatrix([], [], [0] * (m + 1), self.shape)

        elements = list(zip(self.col, self.row, self.data))
        elements.sort()
        
        data = [elem[2] for elem in elements]
        indices = [elem[1] for elem in elements]
        _, m = self.shape

        indptr = [0] * (m + 1)
        for col_idx, _, _ in elements:
            indptr[col_idx + 1] += 1
        
        for j in range(m):
            indptr[j + 1] += indptr[j]
        
        return CSCMatrix(data, indices, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        from CSR import CSRMatrix
        
        if self.nnz == 0:
            n, _ = self.shape
            return CSRMatrix([], [], [0] * (n + 1), self.shape)

        elements = list(zip(self.row, self.col, self.data))
        elements.sort()
        
        data = [elem[2] for elem in elements]
        indices = [elem[1] for elem in elements]
        n, _ = self.shape

        indptr = [0] * (n + 1)
        for row_idx, _, _ in elements:
            indptr[row_idx + 1] += 1
        
        for i in range(n):
            indptr[i + 1] += indptr[i]
        
        return CSRMatrix(data, indices, indptr, self.shape)

    def _to_coo(self) -> 'COOMatrix':
        return self

    def _convert_to_coo(self, other: 'Matrix') -> 'COOMatrix':
        """Вспомогательный метод для преобразования к COO."""
        if isinstance(other, COOMatrix):
            return other
        
        if hasattr(other, '_to_coo'):
            return other._to_coo()

        try:
            other_dense = other.to_dense()
            return COOMatrix.from_dense(other_dense)
        except:
            raise ValueError("Невозможно преобразовать матрицу к COO")