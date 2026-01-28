from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        if len(data) != len(row) or len(data) != len(col) or len(row) != len(col):
            raise ValueError("data, row, col должны быть одинаковой длины")

        sr, sc = shape
        for r in row:
            if r < 0 or r >= sr:
                raise ValueError("row индекс вне диапазона")
        for c in col:
            if c < 0 or c >= sc:
                raise ValueError("col индекс вне диапазона")

        self.data = list(data)
        self.row = list(row)
        self.col = list(col)

        self._normalize()

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу"""
        sr, sc = self.shape
        dm = [[0.0 for _ in range(sc)] for _ in range(sr)]
        for d, r, c in zip(self.data, self.row, self.col):
            dm[r][c] += float(d)
        return dm

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц"""
        if isinstance(other, COOMatrix):
            b = other
        elif hasattr(other, "_to_coo"):
            b = other._to_coo()
        else:
            b = COOMatrix.from_dense(other.to_dense())

        mp = {}
        for d, r, c in zip(self.data, self.row, self.col):
            mp[(r, c)] = mp.get((r, c), 0.0) + float(d)
        for d, r, c in zip(b.data, b.row, b.col):
            mp[(r, c)] = mp.get((r, c), 0.0) + float(d)

        data, row, col = [], [], []
        for (r, c), d in mp.items():
            if d != 0.0:
                row.append(r)
                col.append(c)
                data.append(d)
        return COOMatrix(data, row, col, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр"""
        data = [float(d) * float(scalar) for d in self.data]
        return COOMatrix(data, list(self.row), list(self.col), self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы"""
        sr, sc = self.shape
        return COOMatrix(list(self.data), list(self.col), list(self.row), (sc, sr))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц"""
        csr = self._to_csr()
        res = csr._matmul_impl(other)
        if hasattr(res, "_to_coo"):
            return res._to_coo()
        return COOMatrix.from_dense(res.to_dense())

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы"""
        sr = len(dense_matrix)
        sc = len(dense_matrix[0]) if sr > 0 else 0
        data, row, col = [], [], []
        for r in range(sr):
            if len(dense_matrix[r]) != sc:
                raise ValueError("Плотная матрица должна быть прямоугольной")
            for c in range(sc):
                d = float(dense_matrix[r][c])
                if d != 0.0:
                    data.append(d)
                    row.append(r)
                    col.append(c)
        return cls(data, row, col, (sr, sc))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix

        sr, sc = self.shape

        coo = COOMatrix(list(self.data), list(self.row), list(self.col), self.shape)
        items = list(zip(coo.col, coo.row, coo.data))
        items.sort()

        data: list[float] = []
        i: list[int] = []
        indptr: list[int] = [0] * (sc + 1)

        for c, _, _ in items:
            indptr[c + 1] += 1
        for c in range(sc):
            indptr[c + 1] += indptr[c]

        for c, r, d in items:
            i.append(int(r))
            data.append(float(d))

        return CSCMatrix(data, i, indptr, (sr, sc))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix

        sr, sc = self.shape
        coo = COOMatrix(list(self.data), list(self.row), list(self.col), self.shape)
        items = list(zip(coo.row, coo.col, coo.data))
        items.sort()

        data: list[float] = []
        i: list[int] = []
        indptr: list[int] = [0] * (sr + 1)

        for r, _, _ in items:
            indptr[r + 1] += 1
        for r in range(sr):
            indptr[r + 1] += indptr[r]

        for r, c, d in items:
            i.append(int(c))
            data.append(float(d))

        return CSRMatrix(data, i, indptr, (sr, sc))

    def _normalize(self) -> None:
        mp = {}
        for d, r, c in zip(self.data, self.row, self.col):
            d = float(d)
            if d == 0.0:
                continue
            mp[(int(r), int(c))] = mp.get((int(r), int(c)), 0.0) + d

        items = [((r, c), d) for (r, c), d in mp.items() if d != 0.0]
        items.sort(key=lambda x: (x[0][0], x[0][1]))

        self.row = [r for (r, _), _ in items]
        self.col = [c for (_, c), _ in items]
        self.data = [d for _, d in items]
