from base import Matrix
from matrix_types import COOData, COORows, COOCols, Shape, DenseMatrix


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)

        if not (len(data) == len(row) == len(col)):
            raise ValueError("data, row и col должны быть одинаковой длины")

        acc = {}
        for v, i, j in zip(data, row, col):
            if v == 0:
                continue
            acc[(i, j)] = acc.get((i, j), 0) + v

        self.data = []
        self.row = []
        self.col = []

        for (i, j), v in acc.items():
            if v != 0:
                self.row.append(i)
                self.col.append(j)
                self.data.append(v)

    def to_dense(self) -> DenseMatrix:
        n, m = self.shape
        dense = [[0.0 for _ in range(m)] for _ in range(n)]
        for v, i, j in zip(self.data, self.row, self.col):
            dense[i][j] += v
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        if not isinstance(other, COOMatrix):
            other = other._to_coo()

        return COOMatrix(
            self.data + other.data,
            self.row + other.row,
            self.col + other.col,
            self.shape
        )

    def _mul_impl(self, scalar: float) -> 'Matrix':
        if scalar == 0:
            return COOMatrix([], [], [], self.shape)

        data, row, col = [], [], []
        for v, i, j in zip(self.data, self.row, self.col):
            nv = v * scalar
            if nv != 0:
                data.append(nv)
                row.append(i)
                col.append(j)

        return COOMatrix(data, row, col, self.shape)

    def transpose(self) -> 'Matrix':
        n, m = self.shape
        return COOMatrix(
            self.data.copy(),
            self.col.copy(),
            self.row.copy(),
            (m, n)
        )

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        return self._to_csr() @ other

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        data, row, col = [], [], []
        n = len(dense_matrix)
        m = len(dense_matrix[0]) if n > 0 else 0

        for i in range(n):
            for j in range(m):
                v = dense_matrix[i][j]
                if v != 0:
                    data.append(v)
                    row.append(i)
                    col.append(j)

        return cls(data, row, col, (n, m))

    def _to_csr(self) -> 'CSRMatrix':
        from CSR import CSRMatrix

        n, m = self.shape
        nnz = len(self.data)

        indptr = [0] * (n + 1)
        for i in self.row:
            indptr[i + 1] += 1
        for i in range(n):
            indptr[i + 1] += indptr[i]

        data = [0] * nnz
        indices = [0] * nnz
        counter = indptr.copy()

        for v, i, j in zip(self.data, self.row, self.col):
            pos = counter[i]
            data[pos] = v
            indices[pos] = j
            counter[i] += 1

        return CSRMatrix(data, indices, indptr, self.shape)

    def _to_csc(self) -> 'CSCMatrix':
        from CSC import CSCMatrix

        n, m = self.shape
        nnz = len(self.data)

        indptr = [0] * (m + 1)
        for j in self.col:
            indptr[j + 1] += 1
        for j in range(m):
            indptr[j + 1] += indptr[j]

        data = [0] * nnz
        indices = [0] * nnz
        counter = indptr.copy()

        for v, i, j in zip(self.data, self.row, self.col):
            pos = counter[j]
            data[pos] = v
            indices[pos] = i
            counter[j] += 1

        return CSCMatrix(data, indices, indptr, self.shape)
