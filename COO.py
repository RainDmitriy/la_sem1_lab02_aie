from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        if len(data) != len(row) or len(data) != len(col):
            raise ValueError("Количество элементов в data, row, col должно совпадать")

        self.data = list(data)
        self.row = list(row)
        self.col = list(col)
        self.shape = shape

    def to_dense(self) -> DenseMatrix:
        a, b = self.shape
        m = [[0.0] * b for _ in range(a)]
        for x, i, j in zip(self.data, self.row, self.col):
            m[i][j] += float(x)
        return m

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        if hasattr(other, "_to_coo"):
            o = other._to_coo()
        else:
            o = COOMatrix.from_dense(other.to_dense())

        mp = {}
        for x, i, j in zip(self.data, self.row, self.col):
            mp[(i, j)] = mp.get((i, j), 0.0) + float(x)
        for x, i, j in zip(o.data, o.row, o.col):
            mp[(i, j)] = mp.get((i, j), 0.0) + float(x)

        d, r, c = [], [], []
        for (i, j), x in mp.items():
            if abs(x) > 1e-14:
                d.append(x)
                r.append(i)
                c.append(j)
        return COOMatrix(d, r, c, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        d = [float(x) * float(scalar) for x in self.data]
        return COOMatrix(d, self.row[:], self.col[:], self.shape)

    def transpose(self) -> 'Matrix':
        a, b = self.shape
        return COOMatrix(self.data[:], self.col[:], self.row[:], (b, a))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        if self.shape[1] != other.shape[0]:
            raise ValueError("Несовместимые размеры матриц для умножения")

        a, b = self.shape[0], other.shape[1]
        mp = {}
        if hasattr(other, "_to_csr"):
            o = other._to_csr()
        else:
            o = COOMatrix.from_dense(other.to_dense())._to_csr()

        for p in range(len(self.data)):
            i = self.row[p]
            j = self.col[p]
            x = float(self.data[p])
            s = o.indptr[j]
            e = o.indptr[j + 1]
            for q in range(s, e):
                jj = o.indices[q]
                y = float(o.data[q])
                mp[(i, jj)] = mp.get((i, jj), 0.0) + x * y

        d, r, c = [], [], []
        for (i, j), x in mp.items():
            if abs(x) > 1e-14:
                d.append(x)
                r.append(i)
                c.append(j)
        return COOMatrix(d, r, c, (a, b))

    @classmethod
    def from_dense(cls, dense: DenseMatrix) -> 'COOMatrix':
        d, r, c = [], [], []
        a = len(dense)
        b = len(dense[0]) if a > 0 else 0
        for i in range(a):
            for j in range(b):
                x = float(dense[i][j])
                if x != 0.0:
                    d.append(x)
                    r.append(i)
                    c.append(j)
        return cls(d, r, c, (a, b))

    def _to_csc(self) -> 'CSCMatrix':
        from CSC import CSCMatrix
        a, b = self.shape
        it = list(zip(self.col, self.row, self.data))
        it.sort()
        d = []
        i = []
        p = [0] * (b + 1)
        for j, r, x in it:
            d.append(float(x))
            i.append(int(r))
            p[j + 1] += 1
        for j in range(b):
            p[j + 1] += p[j]
        return CSCMatrix(d, i, p, (a, b))

    def _to_csr(self) -> 'CSRMatrix':
        from CSR import CSRMatrix
        a, b = self.shape
        it = list(zip(self.row, self.col, self.data))
        it.sort()
        d = []
        i = []
        p = [0] * (a + 1)
        for r, c, x in it:
            d.append(float(x))
            i.append(int(c))
            p[r + 1] += 1
        for r in range(a):
            p[r + 1] += p[r]
        return CSRMatrix(d, i, p, (a, b))

    def _to_coo(self) -> 'COOMatrix':
        return COOMatrix(self.data[:], self.row[:], self.col[:], self.shape)