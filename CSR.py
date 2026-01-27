from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix

class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = list(data)
        self.indices = list(indices)
        self.indptr = list(indptr)

    def to_dense(self) -> DenseMatrix:
        rows, cols = self.shape
        res = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for idx in range(self.indptr[i], self.indptr[i+1]):
                res[i][self.indices[idx]] = self.data[idx]
        return res

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        res_dense = self.to_dense()
        other_dense = other.to_dense()
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                res_dense[i][j] += other_dense[i][j]
        return CSRMatrix.from_dense(res_dense)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        if scalar == 0:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)
        return CSRMatrix([v * scalar for v in self.data], self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        from CSC import CSCMatrix
        # При транспонировании CSR логически превращается в CSC
        return CSCMatrix(self.data, self.indices, self.indptr, (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        from COO import COOMatrix
        res_coo = COOMatrix.from_dense(self.to_dense()) @ other
        return CSRMatrix.from_dense(res_coo.to_dense())

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        data, indices, indptr = [], [], [0]
        for i in range(rows):
            for j in range(cols):
                val = dense_matrix[i][j]
                if abs(val) > 1e-15:
                    data.append(val)
                    indices.append(j)
            indptr.append(len(data))
        return cls(data, indices, indptr, (rows, cols))

    def _to_coo(self):
        from COO import COOMatrix
        return COOMatrix.from_dense(self.to_dense())

    def _to_csc(self):
        from CSC import CSCMatrix
        return CSCMatrix.from_dense(self.to_dense())

