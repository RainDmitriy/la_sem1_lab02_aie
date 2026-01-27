from base import Matrix
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix

class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        # Атрибуты строго по условию
        self.data = list(data)
        self.indices = list(indices)
        self.indptr = list(indptr)

    def to_dense(self) -> DenseMatrix:
        r, c = self.shape
        res = [[0.0] * c for _ in range(r)]
        for j in range(c):
            for idx in range(self.indptr[j], self.indptr[j+1]):
                res[self.indices[idx]][j] = self.data[idx]
        return res

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        d, idx, ptr = [], [], [0]
        for j in range(cols):
            for i in range(rows):
                if dense_matrix[i][j] != 0:
                    d.append(dense_matrix[i][j])
                    idx.append(i)
            ptr.append(len(d))
        return cls(d, idx, ptr, (rows, cols))

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        s_dense = self.to_dense()
        o_dense = other.to_dense()
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                s_dense[i][j] += o_dense[i][j]
        return CSCMatrix.from_dense(s_dense)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        if abs(scalar) < 1e-15:
            return CSCMatrix([], [], [0]*(self.shape[1]+1), self.shape)
        return CSCMatrix([v * scalar for v in self.data], self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        from CSR import CSRMatrix
        return CSRMatrix(self.data, self.indices, self.indptr, (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        from COO import COOMatrix
        temp = COOMatrix.from_dense(self.to_dense()) @ other
        return CSCMatrix.from_dense(temp.to_dense())

    def _to_csr(self):
        from CSR import CSRMatrix
        return CSRMatrix.from_dense(self.to_dense())
