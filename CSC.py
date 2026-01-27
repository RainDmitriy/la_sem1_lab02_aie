from base import Matrix
from mytypes import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix
from COO import COOMatrix

ZERO_THRESHOLD = 1e-12
MAX_DENSE_SIZE = 10000


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        n, m = shape
        if len(indptr) != m + 1:
            raise ValueError(f"indptr должен иметь длину m+1 = {m+1}, получено {len(indptr)}")

        for j in range(m):
            if indptr[j] > indptr[j + 1]:
                raise ValueError(f"indptr должен быть неубывающим: indptr[{j}] = {indptr[j]} > indptr[{j+1}] = {indptr[j+1]}")

        if indptr[0] != 0:
            raise ValueError(f"indptr[0] должен быть 0, получено {indptr[0]}")
        if indptr[-1] != len(data):
            raise ValueError(f"indptr[-1] должен быть равен len(data) = {len(data)}, получено {indptr[-1]}")

        if len(data) != len(indices):
            raise ValueError(f"data и indices должны быть одинаковой длины: data={len(data)}, indices={len(indices)}")

        for row_idx in indices:
            if not (0 <= row_idx < n):
                raise ValueError(f"Индекс строки {row_idx} вне диапазона [0, {n-1}]")
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.nnz = len(data)

    def to_dense(self) -> DenseMatrix:
        n, m = self.shape
        if n * m > MAX_DENSE_SIZE:
            raise MemoryError(f"Матрица {n}x{m} слишком большая для dense")
        
        mat = [[0.0] * m for _ in range(n)]
        for j in range(m):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for idx in range(start, end):
                i = self.indices[idx]
                mat[i][j] = self.data[idx]
        return mat

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        self_coo = self._to_coo()
        
        if isinstance(other, CSCMatrix):
            other_coo = other._to_coo()
        else:
            try:
                other_coo = other._to_coo()
            except AttributeError:
                if self.shape[0] * self.shape[1] <= MAX_DENSE_SIZE:
                    other_dense = other.to_dense()
                    other_coo = COOMatrix.from_dense(other_dense)
                else:
                    raise ValueError("Нельзя складывать большие матрицы через dense")
        
        result_coo = self_coo._add_impl(other_coo)
        return result_coo._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        new_data = [x * scalar for x in self.data]
        return CSCMatrix(new_data, self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        from CSR import CSRMatrix

        coo = self._to_coo()
        transposed_coo = COOMatrix(
            coo.data[:],
            coo.col[:],
            coo.row[:],
            (self.shape[1], self.shape[0])
        )
        return transposed_coo._to_csr()

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        self_coo = self._to_coo()
        
        if isinstance(other, CSCMatrix):
            other_coo = other._to_coo()
        elif hasattr(other, '_to_coo'):
            other_coo = other._to_coo()
        else:
            if self.shape[0] * self.shape[1] <= MAX_DENSE_SIZE and \
               other.shape[0] * other.shape[1] <= MAX_DENSE_SIZE:
                other_coo = COOMatrix.from_dense(other.to_dense())
            else:
                raise ValueError("Нельзя умножать большие матрицы через dense")
        
        result_coo = self_coo._matmul_impl(other_coo)
        return result_coo._to_csc()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        return COOMatrix.from_dense(dense_matrix)._to_csc()

    def _to_csr(self) -> 'CSRMatrix':
        from CSR import CSRMatrix
        coo = self._to_coo()
        return coo._to_csr()

    def _to_coo(self) -> 'COOMatrix':
        from COO import COOMatrix
        
        n, m = self.shape
        data, rows, cols = [], [], []
        
        for j in range(m):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for idx in range(start, end):
                data.append(self.data[idx])
                rows.append(self.indices[idx])
                cols.append(j)
        
        return COOMatrix(data, rows, cols, self.shape)