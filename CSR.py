# CSR.py
from base import Matrix
from my_types import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        rows, cols = self.shape
        res = [[0.0 for _ in range(cols)] for _ in range(rows)]
        # по строкам
        for i in range(rows):
            for k in range(self.indptr[i], self.indptr[i + 1]):
                col = self.indices[k]
                val = self.data[k]
                res[i][col] = val
        return res

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        m1 = self.to_dense()
        m2 = other.to_dense()
        rows, cols = self.shape
        res = [[m1[r][c] + m2[r][c] for c in range(cols)] for r in range(rows)]
        return self.from_dense(res)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        # scale
        new_data = [v * scalar for v in self.data]
        return CSRMatrix(new_data, self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        from CSC import CSCMatrix
        # CSR -> CSC
        return CSCMatrix(self.data, self.indices, self.indptr, (self.shape[1], self.shape[0]))

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
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        data, indices, indptr = [], [], [0]
        for r in range(rows):
            for c in range(cols):
                if dense_matrix[r][c] != 0:
                    data.append(dense_matrix[r][c])
                    indices.append(c)
            indptr.append(len(data))
        return cls(data, indices, indptr, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        from CSC import CSCMatrix
        # convert
        return CSCMatrix.from_dense(self.to_dense())

    def _to_coo(self) -> 'COOMatrix':
        from COO import COOMatrix
        rows, cols = self.shape
        r_idx = []
        for i in range(rows):
            for k in range(self.indptr[i], self.indptr[i + 1]):
                r_idx.append(i)
        return COOMatrix(self.data, r_idx, self.indices, self.shape)