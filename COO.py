from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix

class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.data = list(data)
        self.row = list(row)
        self.col = list(col)

    def to_dense(self) -> DenseMatrix:
        rows, cols = self.shape
        grid = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for v, r, c in zip(self.data, self.row, self.col):
            grid[r][c] += v
        return grid

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        res_dense = self.to_dense()
        other_dense = other.to_dense()
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                res_dense[i][j] += other_dense[i][j]
        return COOMatrix.from_dense(res_dense)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        return COOMatrix([v * scalar for v in self.data], self.row, self.col, self.shape)

    def transpose(self) -> 'Matrix':
        return COOMatrix(self.data, self.col, self.row, (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        a_d = self.to_dense()
        b_d = other.to_dense()
        n, k_dim = self.shape
        m = other.shape[1]
        res = [[0.0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for k in range(k_dim):
                if a_d[i][k] == 0: continue
                for j in range(m):
                    res[i][j] += a_d[i][k] * b_d[k][j]
        return COOMatrix.from_dense(res)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        d, r, c = [], [], []
        for i in range(rows):
            for j in range(cols):
                if dense_matrix[i][j] != 0.0:
                    d.append(dense_matrix[i][j])
                    r.append(i)
                    c.append(j)
        return cls(d, r, c, (rows, cols))

    def _to_csc(self):
        from CSC import CSCMatrix
        return CSCMatrix.from_dense(self.to_dense())

    def _to_csr(self):
        from CSR import CSRMatrix
        return CSRMatrix.from_dense(self.to_dense())

    def _to_coo(self):
        return self
