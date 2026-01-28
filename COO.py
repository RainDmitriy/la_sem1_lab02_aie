# COO.py
from base import Matrix
from my_types import COOData, COORows, COOCols, Shape, DenseMatrix
from CSC import CSCMatrix
from CSR import CSRMatrix


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.data = list(data)
        self.row = list(row)
        self.col = list(col)

    def to_dense(self) -> DenseMatrix:
        n, m = self.shape
        dense = [[0.0] * m for _ in range(n)]
        for i in range(len(self.data)):
            dense[self.row[i]][self.col[i]] += self.data[i]
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        if not isinstance(other, COOMatrix):
            other = other._to_coo() if hasattr(other, "_to_coo") else COOMatrix.from_dense(other.to_dense())

        d = {}
        for i in range(len(self.data)):
            key = (self.row[i], self.col[i])
            d[key] = d.get(key, 0.0) + self.data[i]

        for i in range(len(other.data)):
            key = (other.row[i], other.col[i])
            d[key] = d.get(key, 0.0) + other.data[i]

        data, row, col = [], [], []
        for (r, c), v in d.items():
            if v != 0:
                row.append(r)
                col.append(c)
                data.append(v)
        return COOMatrix(data, row, col, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        if scalar == 0:
            return COOMatrix([], [], [], self.shape)
        return COOMatrix([v * scalar for v in self.data], list(self.row), list(self.col), self.shape)

    def transpose(self) -> 'Matrix':
        n, m = self.shape
        return COOMatrix(list(self.data), list(self.col), list(self.row), (m, n))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        if not isinstance(other, COOMatrix):
            other = other._to_coo() if hasattr(other, "_to_coo") else COOMatrix.from_dense(other.to_dense())

        a_n, a_m = self.shape
        b_n, b_m = other.shape

        b_by_row = {}
        for i in range(len(other.data)):
            r, c, v = other.row[i], other.col[i], other.data[i]
            if r not in b_by_row: b_by_row[r] = []
            b_by_row[r].append((c, v))

        acc = {}
        for i in range(len(self.data)):
            ar, ac, av = self.row[i], self.col[i], self.data[i]
            if ac in b_by_row:
                for bc, bv in b_by_row[ac]:
                    idx = (ar, bc)
                    acc[idx] = acc.get(idx, 0.0) + av * bv

        data, row, col = [], [], []
        for (r, c), v in acc.items():
            if v != 0:
                row.append(r)
                col.append(c)
                data.append(v)
        return COOMatrix(data, row, col, (a_n, b_m))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        n = len(dense_matrix)
        m = len(dense_matrix[0]) if n > 0 else 0
        data, row, col = [], [], []
        for i in range(n):
            for j in range(m):
                if dense_matrix[i][j] != 0:
                    row.append(i)
                    col.append(j)
                    data.append(dense_matrix[i][j])
        return cls(data, row, col, (n, m))

    def _to_csc(self) -> 'CSCMatrix':
        n, m = self.shape
        counts = [0] * m
        for c in self.col: counts[c] += 1

        indptr = [0] * (m + 1)
        for j in range(m): indptr[j + 1] = indptr[j] + counts[j]

        next_pos = indptr[:-1].copy()
        data = [0.0] * len(self.data)
        indices = [0] * len(self.data)

        for i in range(len(self.data)):
            v, r, c = self.data[i], self.row[i], self.col[i]
            p = next_pos[c]
            data[p], indices[p] = v, r
            next_pos[c] += 1

        return CSCMatrix(data, indices, indptr, (n, m))

    def _to_csr(self) -> 'CSRMatrix':
        n, m = self.shape
        counts = [0] * n
        for r in self.row: counts[r] += 1

        indptr = [0] * (n + 1)
        for i in range(n): indptr[i + 1] = indptr[i] + counts[i]

        next_pos = indptr[:-1].copy()
        data = [0.0] * len(self.data)
        indices = [0] * len(self.data)

        for i in range(len(self.data)):
            v, r, c = self.data[i], self.row[i], self.col[i]
            p = next_pos[r]
            data[p], indices[p] = v, c
            next_pos[r] += 1

        return CSRMatrix(data, indices, indptr, (n, m))