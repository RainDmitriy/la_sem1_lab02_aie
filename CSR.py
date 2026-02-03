from base import Matrix
from types import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix
import numpy as np
from CSC import CSCMatrix
from COO import COOMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = np.array(data)
        self.indices = np.array(indices)
        self.indptr = np.array(indptr)
        if len(self.indptr) != self.shape[0] + 1:
            raise ValueError("indptr должен иметь shape[0]+1 элементов")

    def to_dense(self) -> DenseMatrix:
        dense = np.zeros(self.shape)
        for row in range(self.shape[0]):
            start = self.indptr[row]
            end = self.indptr[row + 1]
            dense[row, self.indices[start:end]] = self.data[start:end]
        return dense.tolist()

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        dense_self = self.to_dense()
        dense_other = other.to_dense()
        dense_sum = [[dense_self[r][c] + dense_other[r][c] for c in range(len(dense_self[0]))] for r in range(len(dense_self))]
        return COOMatrix.from_dense(dense_sum)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        return CSRMatrix(self.data * scalar, self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        dense_t = self.to_dense()
        dense_t = [[dense_t[c][r] for c in range(self.shape[1])] for r in range(self.shape[0])]
        return CSCMatrix.from_dense(dense_t)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        dense_self = self.to_dense()
        dense_other = other.to_dense()
        rows_self, cols_self = len(dense_self), len(dense_self[0])
        rows_other, cols_other = len(dense_other), len(dense_other[0])
        if cols_self != rows_other:
            raise ValueError("Несовместимые размерности для умножения")
        result = [[sum(dense_self[r][k] * dense_other[k][c] for k in range(cols_self)) for c in range(cols_other)] for r in range(rows_self)]
        return COOMatrix.from_dense(result)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        dense = np.array(dense_matrix)
        rows, cols = np.nonzero(dense)
        data = dense[rows, cols]
        indptr = np.zeros(dense.shape[0] + 1, dtype=int)
        for row in range(dense.shape[0]):
            indptr[row + 1] = indptr[row] + np.sum(rows == row)
        indices = np.zeros(len(data), dtype=int)
        csr_data = np.zeros(len(data))
        pos = 0
        for row in range(dense.shape[0]):
            mask = rows == row
            indices[indptr[row]:indptr[row + 1]] = cols[mask]
            csr_data[indptr[row]:indptr[row + 1]] = data[mask]
            pos = indptr[row + 1]
        return cls(csr_data.tolist(), indices.tolist(), indptr.tolist(), dense.shape)

    def _to_csc(self) -> 'CSCMatrix':
        dense = self.to_dense()
        return CSCMatrix.from_dense(dense)

    def _to_coo(self) -> 'COOMatrix':
        data_list = []
        row_list = []
        col_list = []
        for row in range(self.shape[0]):
            start = self.indptr[row]
            end = self.indptr[row + 1]
            data_list.extend(self.data[start:end])
            row_list.extend([row] * (end - start))
            col_list.extend(self.indices[start:end])
        return COOMatrix(data_list, row_list, col_list, self.shape)
