from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix
from collections import defaultdict


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.row = row
        self.col = col
        if not (len(data) == len(row) == len(col)):
            raise ValueError("Длины data, row и col должны совпадать")

    def to_dense(self) -> DenseMatrix:
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]
        for i in range(len(self.data)):
            r, c, val = self.row[i], self.col[i], self.data[i]
            dense[r][c] = val
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        if not isinstance(other, COOMatrix):
            other_coo = other._to_coo() if hasattr(other, '_to_coo') else COOMatrix.from_dense(other.to_dense())
        else:
            other_coo = other

        rows, cols = self.shape

        result_dict = defaultdict(float)

        for idx in range(len(self.data)):
            key = (self.row[idx], self.col[idx])
            result_dict[key] += self.data[idx]

        for idx in range(len(other_coo.data)):
            key = (other_coo.row[idx], other_coo.col[idx])
            result_dict[key] += other_coo.data[idx]

        data = []
        row_indices = []
        col_indices = []

        sorted_keys = sorted(result_dict.keys(), key=lambda x: (x[0], x[1]))
        for (r, c) in sorted_keys:
            val = result_dict[(r, c)]
            if abs(val) > 1e-12:
                data.append(val)
                row_indices.append(r)
                col_indices.append(c)

        return COOMatrix(data, row_indices, col_indices, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        if abs(scalar) < 1e-14:
            return COOMatrix([], [], [], self.shape)
        new_data = [val * scalar for val in self.data]
        return COOMatrix(new_data, self.row.copy(), self.col.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        return COOMatrix(self.data.copy(), self.col.copy(), self.row.copy(),
                         (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        csr_self = self._to_csr()
        result = csr_self._matmul_impl(other)

        return result._to_coo()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        data = []
        row_indices = []
        col_indices = []

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

        rows, cols = self.shape
        nnz = len(self.data)

        if nnz == 0:
            return CSCMatrix([], [], [0] * (cols + 1), self.shape)

        indices = sorted(range(nnz), key=lambda i: (self.col[i], self.row[i]))

        sorted_data = [self.data[i] for i in indices]
        sorted_rows = [self.row[i] for i in indices]
        sorted_cols = [self.col[i] for i in indices]

        col_counts = [0] * cols
        for j in sorted_cols:
            col_counts[j] += 1

        indptr = [0] * (cols + 1)
        for j in range(cols):
            indptr[j + 1] = indptr[j] + col_counts[j]

        return CSCMatrix(sorted_data, sorted_rows, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        from CSR import CSRMatrix

        rows, cols = self.shape
        nnz = len(self.data)

        if nnz == 0:
            return CSRMatrix([], [], [0] * (rows + 1), self.shape)

        indices = sorted(range(nnz), key=lambda i: (self.row[i], self.col[i]))

        sorted_data = [self.data[i] for i in indices]
        sorted_rows = [self.row[i] for i in indices]
        sorted_cols = [self.col[i] for i in indices]

        row_counts = [0] * rows
        for i in sorted_rows:
            row_counts[i] += 1

        indptr = [0] * (rows + 1)
        for i in range(rows):
            indptr[i + 1] = indptr[i] + row_counts[i]

        return CSRMatrix(sorted_data, sorted_cols, indptr, self.shape)

    @classmethod
    def from_coo(cls, data: COOData, rows: COORows, cols: COOCols, shape: Shape) -> 'COOMatrix':
        return cls(data, rows, cols, shape)