from .base import Matrix
from .types import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        sr, sc = shape
        if len(indptr) != sr + 1:
            raise ValueError("indptr должен иметь длину rows + 1")
        if len(data) != len(indices):
            raise ValueError("data и indices должны быть одинаковой длины")
        if indptr[0] != 0:
            raise ValueError("indptr[0] должен быть 0")
        if indptr[-1] != len(data):
            raise ValueError("indptr[-1] должен совпадать с len(data)")
        for k in indices:
            if k < 0 or k >= sc:
                raise ValueError("indices индекс вне диапазона")

        self.data = [float(x) for x in data]
        self.indices = [int(x) for x in indices]
        self.indptr = [int(x) for x in indptr]

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу"""
        sr, sc = self.shape
        dm = [[0.0 for _ in range(sc)] for _ in range(sr)]
        for r in range(sr):
            s = self.indptr[r]
            e = self.indptr[r + 1]
            for p in range(s, e):
                c = self.indices[p]
                dm[r][c] += self.data[p]
        return dm

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц"""
        coo = self._to_coo()
        res = coo._add_impl(other)
        return res._to_csr()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр"""
        data = [v * float(scalar) for v in self.data]
        return CSRMatrix(data, list(self.indices), list(self.indptr), self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы
        """
        from .CSC import CSCMatrix

        sr, sc = self.shape
        return CSCMatrix(list(self.data), list(self.indices), list(self.indptr), (sc, sr))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц"""
        if hasattr(other, "_to_csr"):
            b = other._to_csr()
        else:
            from .COO import COOMatrix
            b = COOMatrix.from_dense(other.to_dense())._to_csr()

        sr, sc = self.shape
        sc2, sc3 = b.shape
        if sc != sc2:
            raise ValueError("Несовместимые размерности для умножения")

        data: list[float] = []
        i: list[int] = []
        indptr: list[int] = [0]

        for r in range(sr):
            mp: dict[int, float] = {}

            s = self.indptr[r]
            e = self.indptr[r + 1]
            for p in range(s, e):
                k = self.indices[p]
                ad = self.data[p]
                b_s = b.indptr[k]
                b_e = b.indptr[k + 1]
                for bp in range(b_s, b_e):
                    c = b.indices[bp]
                    mp[c] = mp.get(c, 0.0) + ad * b.data[bp]

            cols = sorted(mp.keys())
            for c in cols:
                d = mp[c]
                if d != 0.0:
                    i.append(c)
                    data.append(d)
            indptr.append(len(data))

        return CSRMatrix(data, i, indptr, (sr, sc3))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы"""
        sr = len(dense_matrix)
        sc = len(dense_matrix[0]) if sr > 0 else 0
        data: list[float] = []
        i: list[int] = []
        indptr: list[int] = [0]

        for r in range(sr):
            if len(dense_matrix[r]) != sc:
                raise ValueError("Плотная матрица должна быть прямоугольной")
            for c in range(sc):
                d = float(dense_matrix[r][c])
                if d != 0.0:
                    data.append(d)
                    i.append(c)
            indptr.append(len(data))

        return cls(data, i, indptr, (sr, sc))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSRMatrix в CSCMatrix
        """
        return self._to_coo()._to_csc()
    
    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSRMatrix в COOMatrix
        """
        from .COO import COOMatrix

        sr, _ = self.shape
        data: list[float] = []
        row: list[int] = []
        col: list[int] = []
        for r in range(sr):
            s = self.indptr[r]
            e = self.indptr[r + 1]
            for p in range(s, e):
                row.append(r)
                col.append(self.indices[p])
                data.append(self.data[p])
        return COOMatrix(data, row, col, self.shape)
