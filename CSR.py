from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        if len(indptr) != shape[0] + 1:
            raise ValueError(f"Длина indptr должна быть {shape[0] + 1}")
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
        for i in range(a):
            s = self.indptr[i]
            e = self.indptr[i + 1]
            for p in range(s, e):
                j = self.indices[p]
                m[i][j] = self.data[p]
        return m

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        coo_self = self._to_coo()
        coo_other = other._to_coo()
        coo_sum = coo_self._add_impl(coo_other)
        return coo_sum._to_csr()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        if abs(scalar) < 1e-14:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)

        d = [float(x) * float(scalar) for x in self.data]
        return CSRMatrix(d, self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        from CSC import CSCMatrix
        a, b = self.shape
        return CSCMatrix(self.data[:], self.indices[:], self.indptr[:], (b, a))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        if hasattr(other, "_to_csr"):
            o = other._to_csr()
        else:
            from COO import COOMatrix
            o = COOMatrix.from_dense(other.to_dense())._to_csr()

        a, b = self.shape
        aa, bb = o.shape
        if b != aa:
            raise ValueError("Несовместимые размеры матриц для умножения")

        d = []
        j = []
        p = [0] * (a + 1)

        for i in range(a):
            mp = {}

            s = self.indptr[i]
            e = self.indptr[i + 1]

            for q in range(s, e):
                k = self.indices[q]
                x = self.data[q]

                s2 = o.indptr[k]
                e2 = o.indptr[k + 1]

                for t in range(s2, e2):
                    c = o.indices[t]
                    y = o.data[t]
                    mp[c] = mp.get(c, 0.0) + x * y

            cols = sorted(mp.keys())
            for c in cols:
                x = mp[c]
                if abs(x) > 1e-14:
                    d.append(x)
                    j.append(c)

            p[i + 1] = len(d)

        return CSRMatrix(d, j, p, (a, bb))

    @classmethod
    def from_dense(cls, dense: DenseMatrix) -> 'CSRMatrix':
        a = len(dense)
        b = len(dense[0]) if a > 0 else 0
        d = []
        j = []
        p = [0]
        for i in range(a):
            for k in range(b):
                x = float(dense[i][k])
                if x != 0.0:
                    d.append(x)
                    j.append(k)
            p.append(len(d))
        return cls(d, j, p, (a, b))

    def _to_csc(self) -> 'CSCMatrix':
        from CSC import CSCMatrix
        a, b = self.shape
        cnt = [0] * b
        for c in self.indices:
            cnt[c] += 1
        p = [0] * (b + 1)
        for c in range(b):
            p[c + 1] = p[c] + cnt[c]
        d = [0.0] * len(self.data)
        i = [0] * len(self.indices)
        pos = p.copy()
        for r in range(a):
            s = self.indptr[r]
            e = self.indptr[r + 1]
            for q in range(s, e):
                c = self.indices[q]
                x = self.data[q]
                t = pos[c]
                d[t] = x
                i[t] = r
                pos[c] += 1
        return CSCMatrix(d, i, p, (a, b))

    def _to_coo(self) -> 'COOMatrix':
        from COO import COOMatrix
        a, b = self.shape
        d, r, c = [], [], []
        for i in range(a):
            s = self.indptr[i]
            e = self.indptr[i + 1]
            for p in range(s, e):
                j = self.indices[p]
                x = self.data[p]
                d.append(x)
                r.append(i)
                c.append(j)
        return COOMatrix(d, r, c, (a, b))

    def _to_csr(self) -> 'CSRMatrix':
        return self