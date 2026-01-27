from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.data = data.copy()
        self.row = row.copy()
        self.col = col.copy()
        self.nnz = len(data)

    def to_dense(self) -> DenseMatrix:
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]
        for val, r, c in zip(self.data, self.row, self.col):
            dense[r][c] = val
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        from CSR import CSRMatrix

        if not isinstance(self, CSRMatrix):
            self_csr = self._to_csr()
        else:
            self_csr = self

        if not isinstance(other, CSRMatrix):
            other_csr = CSRMatrix.from_dense(other.to_dense())
        else:
            other_csr = other

        result_csr = self_csr._add_impl(other_csr)

        return result_csr._to_coo()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        if scalar == 0:
            return COOMatrix([], [], [], self.shape)
        new_data = [val * scalar for val in self.data]
        return COOMatrix(new_data, self.row.copy(), self.col.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        return COOMatrix(self.data.copy(), self.col.copy(), self.row.copy(),
                         (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        from CSR import CSRMatrix

        if not isinstance(self, CSRMatrix):
            self_csr = self._to_csr()
        else:
            self_csr = self

        if not isinstance(other, CSRMatrix):
            other_csr = CSRMatrix.from_dense(other.to_dense())
        else:
            other_csr = other

        result_csr = self_csr._matmul_impl(other_csr)

        return result_csr._to_coo()

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
                if val != 0:
                    data.append(val)
                    row_indices.append(i)
                    col_indices.append(j)

        return cls(data, row_indices, col_indices, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        from CSC import CSCMatrix

        if self.nnz == 0:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)

        indices_by_col = sorted(range(self.nnz), key=lambda i: self.col[i])

        data = []
        indices = []
        indptr = [0] * (self.shape[1] + 1)

        current_col = 0
        for idx in indices_by_col:
            col = self.col[idx]
            while current_col < col:
                indptr[current_col + 1] = len(data)
                current_col += 1
            data.append(self.data[idx])
            indices.append(self.row[idx])

        while current_col < self.shape[1]:
            indptr[current_col + 1] = len(data)
            current_col += 1

        return CSCMatrix(data, indices, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        from CSR import CSRMatrix

        if self.nnz == 0:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)

        indices_by_row = sorted(range(self.nnz), key=lambda i: self.row[i])

        data = []
        indices = []
        indptr = [0] * (self.shape[0] + 1)

        current_row = 0
        for idx in indices_by_row:
            row = self.row[idx]
            while current_row < row:
                indptr[current_row + 1] = len(data)
                current_row += 1
            data.append(self.data[idx])
            indices.append(self.col[idx])

        while current_row < self.shape[0]:
            indptr[current_row + 1] = len(data)
            current_row += 1

        return CSRMatrix(data, indices, indptr, self.shape)