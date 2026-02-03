from base import Matrix
from types import COOData, COORows, COOCols, Shape, DenseMatrix
from CSR import CSRMatrix
from CSC import CSCMatrix
import numpy as np

class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        if len(data) != len(row) or len(data) != len(col):
            raise ValueError("Несовместимые длины списков data, row, col")
        if max(row) >= shape[0] or max(col) >= shape[1]:
            raise ValueError("Индексы выходят за границы shape")
        self.data = np.array(data)
        self.row = np.array(row)
        self.col = np.array(col)

    def to_dense(self) -> DenseMatrix:
        dense = np.zeros(self.shape)
        for i in range(len(self.data)):
            dense[self.row[i], self.col[i]] = self.data[i]
        return dense.tolist()

    @classmethod
    def from_dense(cls, dense: DenseMatrix) -> 'COOMatrix':
        data_list, row_list, col_list = [], [], []
        for r in range(len(dense)):
            for c in range(len(dense[r])):
                if dense[r][c] != 0:
                    data_list.append(dense[r][c])
                    row_list.append(r)
                    col_list.append(c)
        return cls(data_list, row_list, col_list, (len(dense), len(dense[0])))

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        return COOMatrix.from_dense(self.to_dense() + other.to_dense())

    def _mul_impl(self, scalar: float) -> 'Matrix':
        return COOMatrix(self.data * scalar, self.row, self.col, self.shape)

    def transpose(self) -> 'Matrix':
        return COOMatrix(self.data, self.col, self.row, (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        return COOMatrix.from_dense(np.dot(self.to_dense(), other.to_dense()))

    def _to_csr(self) -> CSRMatrix:
        dense = self.to_dense()
        data, indices, indptr = [], [], [0]
        for r in range(self.shape[0]):
            row_nonzero = [(c, dense[r][c]) for c in range(self.shape[1]) if dense[r][c] != 0]
            for c, val in row_nonzero:
                data.append(val)
                indices.append(c)
            indptr.append(indptr[-1] + len(row_nonzero))
        return CSRMatrix(data, indices, indptr, self.shape)

    def _to_csc(self) -> CSCMatrix:
        dense = self.to_dense()
        data, indices, indptr = [], [], [0]
        for c in range(self.shape[1]):
            col_nonzero = [(r, dense[r][c]) for r in range(self.shape[0]) if dense[r][c] != 0]
            for r, val in col_nonzero:
                data.append(val)
                indices.append(r)
            indptr.append(indptr[-1] + len(col_nonzero))
        return CSCMatrix(data, indices, indptr, self.shape)
