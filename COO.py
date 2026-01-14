from base import Matrix
from types import COOData, COORows, COOCols, Shape, DenseMatrix
from CSC import CSCMatrix
from CSR import CSRMatrix

class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.data = list(data)
        self.row = list(row)
        self.col = list(col)

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу."""
        n, m = self.shape
        dense = [[0.0 for _ in range(m)] for _ in range(n)]
        for v, r, c in zip(self.data, self.row, self.col):
            dense[r][c] += v
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц."""
        if not isinstance(other, COOMatrix):
            if hasattr(other, "_to_coo"):
                other = other._to_coo()
            else:
                other = COOMatrix.from_dense(other.to_dense())

        d = {}
        for v, r, c in zip(self.data, self.row, self.col):
            d[(r, c)] = d.get((r, c), 0.0) + v
        for v, r, c in zip(other.data, other.row, other.col):
            d[(r, c)] = d.get((r, c), 0.0) + v

        data, row, col = [], [], []
        for (r, c), v in d.items():
            if v != 0:
                row.append(r)
                col.append(c)
                data.append(v)
        return COOMatrix(data, row, col, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        if scalar == 0: return COOMatrix([], [], [], self.shape)
        data = [v * scalar for v in self.data]
        return COOMatrix(data, list(self.row), list(self.col), self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        n, m = self.shape
        return COOMatrix(list(self.data), list(self.col), list(self.row), (m, n))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""
        if not isinstance(other, COOMatrix):
            if hasattr(other, "_to_coo"):
                other = other._to_coo()
            else:
                other = COOMatrix.from_dense(other.to_dense())

        a_n, a_m = self.shape
        b_n, b_m = other.shape

        b_by_row = {}
        for v, r, c in zip(other.data, other.row, other.col):
            b_by_row.setdefault(r, []).append((c, v))

        acc = {}
        for av, ar, ac in zip(self.data, self.row, self.col):
            if ac not in b_by_row:
                continue
            for bc, bv in b_by_row[ac]:
                key = (ar, bc)
                acc[key] = acc.get(key, 0.0) + av * bv

        data, row, col = [], [], []
        for (r, c), v in acc.items():
            if v != 0:
                row.append(r)
                col.append(c)
                data.append(v)
        return COOMatrix(data, row, col, (a_n, b_m))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        n = len(dense_matrix)
        m = len(dense_matrix[0]) if n > 0 else 0
        data, row, col = [], [], []
        for i in range(n):
            for j in range(m):
                v = dense_matrix[i][j]
                if v != 0:
                    row.append(i)
                    col.append(j)
                    data.append(v)
        return cls(data, row, col, (n, m))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        n, m = self.shape
        nnz = len(self.data)

        counts = [0] * m
        for c in self.col:
            counts[c] += 1

        indptr = [0] * (m + 1)
        for j in range(m):
            indptr[j + 1] = indptr[j] + counts[j]

        next_pos = indptr[:-1].copy()
        data = [0.0] * nnz
        indices = [0] * nnz

        for v, r, c in zip(self.data, self.row, self.col):
            p = next_pos[c]
            data[p] = v
            indices[p] = r
            next_pos[c] += 1

        return CSCMatrix(data, indices, indptr, (n, m))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        n, m = self.shape
        nnz = len(self.data)

        counts = [0] * n
        for r in self.row:
            counts[r] += 1

        indptr = [0] * (n + 1)
        for i in range(n):
            indptr[i + 1] = indptr[i] + counts[i]

        next_pos = indptr[:-1].copy()
        data = [0.0] * nnz
        indices = [0] * nnz

        for v, r, c in zip(self.data, self.row, self.col):
            p = next_pos[r]
            data[p] = v
            indices[p] = c
            next_pos[r] += 1

        return CSRMatrix(data, indices, indptr, (n, m))
