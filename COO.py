# COO.py
from base import Matrix
from my_types import COOData, COORows, COOCols, Shape, DenseMatrix

class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.row = row
        self.col = col

    def to_dense(self) -> DenseMatrix:
        rows, cols = self.shape
        res = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for i in range(len(self.data)):
            res[self.row[i]][self.col[i]] = self.data[i]
        return res

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        m1 = self.to_dense()
        m2 = other.to_dense()
        rows, cols = self.shape
        res = [[m1[r][c] + m2[r][c] for c in range(cols)] for r in range(rows)]
        return self.from_dense(res)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        new_data = [val * scalar for val in self.data]
        return COOMatrix(new_data, self.row, self.col, self.shape)

    def transpose(self) -> 'Matrix':
        # swap
        return COOMatrix(self.data, self.col, self.row, (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        m1 = self.to_dense()
        m2 = other.to_dense()
        r1, c1 = self.shape
        c2 = other.shape[1]
        res = [[0.0 for _ in range(c2)] for _ in range(r1)]
        for i in range(r1):
            for j in range(c2):
                for k in range(c1):
                    res[i][j] += m1[i][k] * m2[k][j]
        return self.from_dense(res)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        data, r_idx, c_idx = [], [], []
        for r in range(rows):
            for c in range(cols):
                if dense_matrix[r][c] != 0:
                    data.append(dense_matrix[r][c])
                    r_idx.append(r)
                    c_idx.append(c)
        return cls(data, r_idx, c_idx, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        from CSC import CSCMatrix
        # convert
        return CSCMatrix.from_dense(self.to_dense())

    def _to_csr(self) -> 'CSRMatrix':
        from CSR import CSRMatrix
        # convert
        return CSRMatrix.from_dense(self.to_dense())