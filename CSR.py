from base import Matrix
from matrix_types import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)

        if len(indptr) != shape[0] + 1:
            raise ValueError("Некорректная длина indptr")

        if len(data) != len(indices):
            raise ValueError("data и indices должны быть одной длины")

        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        n, m = self.shape
        dense = [[0.0 for _ in range(m)] for _ in range(n)]

        for i in range(n):
            start, end = self.indptr[i], self.indptr[i + 1]
            for idx in range(start, end):
                j = self.indices[idx]
                dense[i][j] += self.data[idx]

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        if not isinstance(other, CSRMatrix):
            other = CSRMatrix.from_dense(other.to_dense())

        from COO import COOMatrix
        return (self._to_coo() + other._to_coo())._to_csr()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        if scalar == 0:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)

        data = []
        indices = []
        indptr = [0]

        for i in range(self.shape[0]):
            start, end = self.indptr[i], self.indptr[i + 1]
            for k in range(start, end):
                v = self.data[k] * scalar
                if v != 0:
                    data.append(v)
                    indices.append(self.indices[k])
            indptr.append(len(data))

        return CSRMatrix(data, indices, indptr, self.shape)

    def transpose(self) -> 'Matrix':
        return self._to_csc()

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        if not hasattr(other, '_to_csc'):
            other = other.to_dense()
            from COO import COOMatrix
            other = COOMatrix.from_dense(other)._to_csc()
        else:
            other = other._to_csc()

        n, m = self.shape[0], other.shape[1]
        data = []
        indices = []
        indptr = [0]

        for i in range(n):
            row_result = {}
            a_start, a_end = self.indptr[i], self.indptr[i + 1]

            for a_idx in range(a_start, a_end):
                k = self.indices[a_idx]
                v1 = self.data[a_idx]

                b_start, b_end = other.indptr[k], other.indptr[k + 1]
                for b_idx in range(b_start, b_end):
                    j = other.indices[b_idx]
                    v2 = other.data[b_idx]
                    row_result[j] = row_result.get(j, 0) + v1 * v2

            for j, v in row_result.items():
                if v != 0:
                    data.append(v)
                    indices.append(j)

            indptr.append(len(data))

        return CSRMatrix(data, indices, indptr, (n, m))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        data = []
        indices = []
        indptr = [0]

        n = len(dense_matrix)
        m = len(dense_matrix[0]) if n > 0 else 0

        for i in range(n):
            for j in range(m):
                v = dense_matrix[i][j]
                if v != 0:
                    data.append(v)
                    indices.append(j)
            indptr.append(len(data))

        return cls(data, indices, indptr, (n, m))

    def _to_csc(self) -> 'CSCMatrix':
        from CSC import CSCMatrix

        n, m = self.shape
        nnz = len(self.data)

        indptr = [0] * (m + 1)
        for j in self.indices:
            indptr[j + 1] += 1

        for j in range(m):
            indptr[j + 1] += indptr[j]

        data = [0] * nnz
        indices = [0] * nnz
        counter = indptr.copy()

        for i in range(n):
            start, end = self.indptr[i], self.indptr[i + 1]
            for idx in range(start, end):
                j = self.indices[idx]
                pos = counter[j]
                data[pos] = self.data[idx]
                indices[pos] = i
                counter[j] += 1

        return CSCMatrix(data, indices, indptr, self.shape)

    def _to_coo(self) -> 'COOMatrix':
        from COO import COOMatrix

        data = []
        row = []
        col = []

        for i in range(self.shape[0]):
            start, end = self.indptr[i], self.indptr[i + 1]
            for idx in range(start, end):
                row.append(i)
                col.append(self.indices[idx])
                data.append(self.data[idx])

        return COOMatrix(data, row, col, self.shape)
