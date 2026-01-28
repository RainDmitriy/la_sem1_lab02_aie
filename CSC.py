from base import Matrix
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        if len(indptr) != shape[1] + 1:
            raise ValueError(f"Длина indptr должна быть {shape[1] + 1}")
        if indptr[0] != 0:
            raise ValueError("Первый элемент indptr должен быть равен 0")
        if indptr[-1] != len(data):
            raise ValueError(f"Последний элемент indptr должен быть равен {len(data)}")
        if len(data) != len(indices):
            raise ValueError("Длины data и indices должны совпадать")

        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.shape = shape

    def to_dense(self) -> DenseMatrix:
        a, b = self.shape
        m = [[0.0] * b for _ in range(a)]
        for j in range(b):
            s = self.indptr[j]
            e = self.indptr[j + 1]
            for p in range(s, e):
                i = self.indices[p]
                m[i][j] = self.data[p]
        return m

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        coo_self = self._to_coo()
        coo_other = other._to_coo()
        coo_sum = coo_self._add_impl(coo_other)
        return coo_sum._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        if abs(scalar) < 1e-14:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)

        d = [float(x) * float(scalar) for x in self.data]
        return CSCMatrix(d, self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        from CSR import CSRMatrix
        a, b = self.shape
        return CSRMatrix(self.data[:], self.indices[:], self.indptr[:], (b, a))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        a = self._to_csr()
        c = a._matmul_impl(other)
        if hasattr(c, "_to_csc"):
            return c._to_csc()
        from COO import COOMatrix
        return COOMatrix.from_dense(c.to_dense())._to_csc()

    @classmethod
    def from_dense(cls, dense: DenseMatrix) -> 'CSCMatrix':
        a = len(dense)
        b = len(dense[0]) if a > 0 else 0
        d = []
        i = []
        cnt = [0] * b
        for j in range(b):
            for r in range(a):
                x = float(dense[r][j])
                if x != 0.0:
                    d.append(x)
                    i.append(r)
                    cnt[j] += 1
        p = [0] * (b + 1)
        for j in range(b):
            p[j + 1] = p[j] + cnt[j]
        return cls(d, i, p, (a, b))

    def _to_csr(self) -> 'CSRMatrix':
        from CSR import CSRMatrix
        a, b = self.shape
        cnt = [0] * a
        for r in self.indices:
            cnt[r] += 1
        p = [0] * (a + 1)
        for r in range(a):
            p[r + 1] = p[r] + cnt[r]
        d = [0.0] * len(self.data)
        i = [0] * len(self.indices)
        pos = p.copy()
        for c in range(b):
            s = self.indptr[c]
            e = self.indptr[c + 1]
            for q in range(s, e):
                r = self.indices[q]
                x = self.data[q]
                t = pos[r]
                d[t] = x
                i[t] = c
                pos[r] += 1
        return CSRMatrix(d, i, p, (a, b))

    def _to_coo(self) -> 'COOMatrix':
        from COO import COOMatrix
        a, b = self.shape
        d, r, c = [], [], []
        for j in range(b):
            s = self.indptr[j]
            e = self.indptr[j + 1]
            for p in range(s, e):
                i = self.indices[p]
                x = self.data[p]
                d.append(x)
                r.append(i)
                c.append(j)
        return COOMatrix(d, r, c, (a, b))

    def _to_csc(self) -> 'CSCMatrix':
        return self