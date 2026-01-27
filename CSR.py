from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix

class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        # Атрибуты строго по условию
        self.data = list(data)
        self.indices = list(indices)
        self.indptr = list(indptr)

    def to_dense(self) -> DenseMatrix:
        r, c = self.shape
        res = [[0.0] * c for _ in range(r)]
        for i in range(r):
            for idx in range(self.indptr[i], self.indptr[i+1]):
                res[i][self.indices[idx]] = self.data[idx]
        return res

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        d, idx, ptr = [], [], [0]
        for row in dense_matrix:
            for j, val in enumerate(row):
                if val != 0:
                    d.append(val)
                    idx.append(j)
            ptr.append(len(d))
        return cls(d, idx, ptr, (rows, cols))

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        d1, d2 = self.to_dense(), other.to_dense()
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                d1[i][j] += d2[i][j]
        return CSRMatrix.from_dense(d1)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        if abs(scalar) < 1e-15:
            return CSRMatrix([], [], [0]*(self.shape[0]+1), self.shape)
        return CSRMatrix([x * scalar for x in self.data], self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        from CSC import CSCMatrix
        # Транспонированный CSR имеет ту же структуру, что CSC
        return CSCMatrix(self.data, self.indices, self.indptr, (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        from COO import COOMatrix
        # Используем COO для промежуточного вычисления
        temp = COOMatrix.from_dense(self.to_dense()) @ other
        return CSRMatrix.from_dense(temp.to_dense())

    def _to_csc(self):
        from CSC import CSCMatrix
        return CSCMatrix.from_dense(self.to_dense())
