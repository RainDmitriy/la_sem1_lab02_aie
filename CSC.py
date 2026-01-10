from base import Matrix
from types import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix
import numpy as np
from CSR import CSRMatrix
from COO import COOMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = np.array(data)
        self.indices = np.array(indices)
        self.indptr = np.array(indptr)
        if len(self.indptr) != self.shape[1] + 1:
            raise ValueError("indptr должен иметь shape[1]+1 элементов")

    def to_dense(self) -> DenseMatrix:
        dense = np.zeros(self.shape)
        for col in range(self.shape[1]):
            start = self.indptr[col]
            end = self.indptr[col + 1]
            dense[self.indices[start:end], col] = self.data[start:end]
        return dense.tolist()

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        dense_self = self.to_dense()
        dense_other = other.to_dense()
        dense_sum = [[dense_self[r][c] + dense_other[r][c] for c in range(len(dense_self[0]))] for r in range(len(dense_self))]
        return COOMatrix.from_dense(dense_sum)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        return CSCMatrix(self.data * scalar, self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        dense_t = self.to_dense()
        dense_t = [[dense_t[c][r] for c in range(self.shape[1])] for r in range(self.shape[0])]
        return CSRMatrix.from_dense(dense_t)

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
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        dense = np.array(dense_matrix)
        rows, cols = np.nonzero(dense)
        data = dense[rows, cols]
        indptr = np.zeros(dense.shape[1] + 1, dtype=int)
        for col in range(dense.shape[1]):
            indptr[col + 1] = indptr[col] + np.sum(cols == col)
        indices = np.zeros(len(data), dtype=int)
        csc_data = np.zeros(len(data))
        pos = 0
        for col in range(dense.shape[1]):
            mask = cols == col
            indices[indptr[col]:indptr[col + 1]] = rows[mask]
            csc_data[indptr[col]:indptr[col + 1]] = data[mask]
            pos = indptr[col + 1]
        return cls(csc_data.tolist(), indices.tolist(), indptr.tolist(), dense.shape)

    def _to_csr(self) -> 'CSRMatrix':
        dense = self.to_dense()
        return CSRMatrix.from_dense(dense)

    def _to_coo(self) -> 'COOMatrix':
        data_list = []
        row_list = []
        col_list = []
        for col in range(self.shape[1]):
            start = self.indptr[col]
            end = self.indptr[col + 1]
            data_list.extend(self.data[start:end])
            row_list.extend(self.indices[start:end])
            col_list.extend([col] * (end - start))
        return COOMatrix(data_list, row_list, col_list, self.shape)
