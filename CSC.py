# CSC.py
from base import Matrix
from my_types import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        rows, cols = self.shape
        res = [[0.0 for _ in range(cols)] for _ in range(rows)]
        # по столбцам
        for j in range(cols):
            for k in range(self.indptr[j], self.indptr[j+1]):
                row = self.indices[k]
                res[row][j] = self.data[k]
        return res

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        m1 = self.to_dense()
        m2 = other.to_dense()
        r, c = self.shape
        res = [[m1[i][j] + m2[i][j] for j in range(c)] for i in range(r)]
        return self.from_dense(res)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        # scale
        new_data = [v * scalar for v in self.data]
        return CSCMatrix(new_data, self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        from CSR import CSRMatrix
        # CSC -> CSR
        return CSRMatrix(self.data, self.indices, self.indptr, (self.shape[1], self.shape[0]))

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
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        data, indices, indptr = [], [], [0]
        # внешний цикл по столбцам
        for j in range(cols):
            for i in range(rows):
                if dense_matrix[i][j] != 0:
                    data.append(dense_matrix[i][j])
                    indices.append(i)
            indptr.append(len(data))
        return cls(data, indices, indptr, (rows, cols))

    def _to_csr(self) -> 'CSRMatrix':
        from CSR import CSRMatrix
        # convert
        return CSRMatrix.from_dense(self.to_dense())

    def _to_coo(self) -> 'COOMatrix':
        from COO import COOMatrix
        rows, cols = self.shape
        c_idx = []
        for j in range(cols):
            for k in range(self.indptr[j], self.indptr[j+1]):
                c_idx.append(j)
        return COOMatrix(self.data, self.indices, c_idx, self.shape)