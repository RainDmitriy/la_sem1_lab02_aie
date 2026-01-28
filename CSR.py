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
        for i in range(rows):
            for k in range(self.indptr[i], self.indptr[i+1]):
                col = self.indices[k]
                res[i][col] = self.data[k]
        return res

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        # Сложение через COO (быстрее для разреженных данных)
        m1_coo = self._to_coo()
        m2_coo = other._to_coo()
        res_coo = m1_coo._add_impl(m2_coo)
        return self.from_dense(res_coo.to_dense())

    def _mul_impl(self, scalar: float) -> 'Matrix':
        new_data = [v * scalar for v in self.data]
        return CSRMatrix(new_data, self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        from CSC import CSCMatrix
        # CSR -> CSC (просто меняем тип, структура данных та же)
        return CSCMatrix(self.data, self.indices, self.indptr, (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        # Умножение через COO без создания плотных матриц
        m1_coo = self._to_coo()
        m2_coo = other._to_coo()
        res_coo = m1_coo._matmul_impl(m2_coo)
        return self.from_dense(res_coo.to_dense())

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
        # Через COO конвертация надежнее
        return self._to_coo()._to_csc()

    def _to_coo(self) -> 'COOMatrix':
        from COO import COOMatrix
        rows, cols = self.shape
        r_idx = []
        for i in range(rows):
            for k in range(self.indptr[i], self.indptr[i+1]):
                r_idx.append(i)
        return COOMatrix(self.data, self.indices, r_idx, self.shape)