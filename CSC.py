from base import Matrix
from matrix_types import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)

        if len(indptr) != shape[1] + 1:
            raise ValueError("Некорректная длина indptr")

        if len(data) != len(indices):
            raise ValueError("data и indices должны быть одной длины")

        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        n, m = self.shape
        dense = [[0.0 for _ in range(m)] for _ in range(n)]

        for j in range(m):
            start, end = self.indptr[j], self.indptr[j + 1]
            for idx in range(start, end):
                i = self.indices[idx]
                dense[i][j] += self.data[idx]

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        if not isinstance(other, CSCMatrix):
            other = other._to_csc()

        from COO import COOMatrix
        return (self._to_coo() + other._to_coo())._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        if scalar == 0:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)

        data = []
        indices = []
        indptr = [0]

        for j in range(self.shape[1]):
            start, end = self.indptr[j], self.indptr[j + 1]
            for k in range(start, end):
                v = self.data[k] * scalar
                if v != 0:
                    data.append(v)
                    indices.append(self.indices[k])
            indptr.append(len(data))

        return CSCMatrix(data, indices, indptr, self.shape)

    def transpose(self) -> 'Matrix':
        return self._to_csr()

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        # CSC @ CSR -> CSC
        if not hasattr(other, '_to_csr'):
            other = other._to_csr()
        else:
            other = other._to_csr()

        from CSR import CSRMatrix

        n, m = self.shape[0], other.shape[1]
        data = []
        indices = []
        indptr = [0]

        # идём по столбцам результата
        for j in range(m):
            col_result = {}
            b_start, b_end = other.indptr[j], other.indptr[j + 1]

            for b_idx in range(b_start, b_end):
                k = other.indices[b_idx]
                v2 = other.data[b_idx]

                a_start, a_end = self.indptr[k], self.indptr[k + 1]
                for a_idx in range(a_start, a_end):
                    i = self.indices[a_idx]
                    v1 = self.data[a_idx]
                    col_result[i] = col_result.get(i, 0) + v1 * v2

            for i, v in col_result.items():
                if v != 0:
                    data.append(v)
                    indices.append(i)

            indptr.append(len(data))

        return CSCMatrix(data, indices, indptr, (n, m))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        data = []
        indices = []
        indptr = [0]

        n = len(dense_matrix)
        m = len(dense_matrix[0]) if n > 0 else 0

        for j in range(m):
            for i in range(n):
                v = dense_matrix[i][j]
                if v != 0:
                    data.append(v)
                    indices.append(i)
            indptr.append(len(data))

        return cls(data, indices, indptr, (n, m))

    def _to_csr(self) -> 'CSRMatrix':
        from CSR import CSRMatrix

        n, m = self.shape
        nnz = len(self.data)

        indptr = [0] * (n + 1)
        for i in self.indices:
            indptr[i + 1] += 1

        for i in range(n):
            indptr[i + 1] += indptr[i]

        data = [0] * nnz
        indices = [0] * nnz
        counter = indptr.copy()

        for j in range(m):
            start, end = self.indptr[j], self.indptr[j + 1]
            for idx in range(start, end):
                i = self.indices[idx]
                pos = counter[i]
                data[pos] = self.data[idx]
                indices[pos] = j
                counter[i] += 1

        return CSRMatrix(data, indices, indptr, self.shape)

    def _to_coo(self) -> 'COOMatrix':
        from COO import COOMatrix

        data = []
        row = []
        col = []

        for j in range(self.shape[1]):
            start, end = self.indptr[j], self.indptr[j + 1]
            for idx in range(start, end):
                row.append(self.indices[idx])
                col.append(j)
                data.append(self.data[idx])

        return COOMatrix(data, row, col, self.shape)
