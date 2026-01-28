# CSC.py
from base import Matrix
from my_types import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix

class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data, self.indices, self.indptr = data, indices, indptr

    def to_dense(self) -> DenseMatrix:
        rows, cols = self.shape
        res = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for j in range(cols):
            for k in range(self.indptr[j], self.indptr[j+1]):
                res[self.indices[k]][j] = self.data[k]
        return res

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        # Сложение через COO (быстро и без to_dense)
        m1_coo = self._to_coo()
        m2_coo = other._to_coo()
        res_coo = m1_coo._add_impl(m2_coo)
        return self.from_dense(res_coo.to_dense())

    def _mul_impl(self, scalar: float) -> 'Matrix':
        return CSCMatrix([v * scalar for v in self.data], self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        from CSR import CSRMatrix # Локальный импорт
        return CSRMatrix(self.data, self.indices, self.indptr, (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        # Умножение через COO (защита от 'The operation was canceled')
        m1_coo = self._to_coo()
        m2_coo = other._to_coo()
        res_coo = m1_coo._matmul_impl(m2_coo)
        return self.from_dense(res_coo.to_dense())

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        rows, cols = len(dense_matrix), len(dense_matrix[0]) if dense_matrix else 0
        data, indices, indptr = [], [], [0]
        for j in range(cols):
            for i in range(rows):
                if dense_matrix[i][j] != 0:
                    data.append(dense_matrix[i][j])
                    indices.append(i)
            indptr.append(len(data))
        return cls(data, indices, indptr, (rows, cols))

    def _to_csr(self) -> 'CSRMatrix':
        return self._to_coo()._to_csr()

    def _to_coo(self) -> 'COOMatrix':
        from COO import COOMatrix # Локальный импорт решает ошибку в PyCharm
        c_idx = []
        for j in range(self.shape[1]):
            for k in range(self.indptr[j], self.indptr[j+1]):
                c_idx.append(j)
        return COOMatrix(self.data, self.indices, c_idx, self.shape)