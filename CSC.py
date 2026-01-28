from base import Matrix
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix

# используется для корректной проверки типов
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from COO import COOMatrix
    from CSR import CSRMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)

        # проверки корректности структуры CSC
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
        """Преобразует CSC‑матрицу в плотный формат."""
        sr, sc = self.shape
        dm = [[0.0] * sc for _ in range(sr)]

        for c in range(sc):
            s = self.indptr[c]
            e = self.indptr[c + 1]
            for p in range(s, e):
                r = self.indices[p]
                d = self.data[p]
                dm[r][c] = d

        return dm

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение двух CSC‑матриц через промежуточное COO‑представление."""
        coo_self = self._to_coo()
        coo_other = other._to_coo()
        coo_sum = coo_self._add_impl(coo_other)
        return coo_sum._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение матрицы на скаляр."""
        if abs(scalar) < 1e-14:
            # результат – нулевая матрица той же размерности
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)

        data = [float(d) * float(scalar) for d in self.data]
        return CSCMatrix(data, self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование матрицы. Результат возвращается в формате CSR."""
        from CSR import CSRMatrix
        sr, sc = self.shape
        # по подсказке можно просто переинтерпретировать массивы
        return CSRMatrix(self.data[:], self.indices[:], self.indptr[:], (sc, sr))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC‑матрицы на другую матрицу."""
        # проще/надёжнее: CSC -> CSR, умножение в CSR, потом обратно
        a = self._to_csr()
        c = a._matmul_impl(other)
        if hasattr(c, "_to_csc"):
            return c._to_csc()
        from COO import COOMatrix
        return COOMatrix.from_dense(c.to_dense())._to_csc()

    @classmethod
    def from_dense(cls, dense: DenseMatrix) -> 'CSCMatrix':
        """Создаёт CSC‑матрицу из плотного представления."""
        sr = len(dense)
        sc = len(dense[0]) if sr > 0 else 0

        data = []
        i = []
        cnt = [0] * sc

        for c in range(sc):
            for r in range(sr):
                d = float(dense[r][c])
                if d != 0.0:
                    data.append(d)
                    i.append(r)
                    cnt[c] += 1

        indptr = [0] * (sc + 1)
        for c in range(sc):
            indptr[c + 1] = indptr[c] + cnt[c]

        return cls(data, i, indptr, (sr, sc))

    def _to_csr(self) -> 'CSRMatrix':
        """Конвертирует CSC‑матрицу в формат CSR."""
        from CSR import CSRMatrix
        sr, sc = self.shape

        # подсчёт количества элементов в каждой строке
        cnt = [0] * sr
        for r in self.indices:
            cnt[r] += 1

        indptr = [0] * (sr + 1)
        for r in range(sr):
            indptr[r + 1] = indptr[r] + cnt[r]

        data = [0.0] * len(self.data)
        i = [0] * len(self.indices)
        pos = indptr.copy()

        for c in range(sc):
            s = self.indptr[c]
            e = self.indptr[c + 1]
            for p in range(s, e):
                r = self.indices[p]
                d = self.data[p]

                p2 = pos[r]
                data[p2] = d
                i[p2] = c
                pos[r] += 1

        return CSRMatrix(data, i, indptr, (sr, sc))

    def _to_coo(self) -> 'COOMatrix':
        """Конвертирует CSC‑матрицу в формат COO."""
        from COO import COOMatrix
        sr, sc = self.shape

        data = []
        row = []
        col = []

        for c in range(sc):
            s = self.indptr[c]
            e = self.indptr[c + 1]
            for p in range(s, e):
                r = self.indices[p]
                d = self.data[p]
                data.append(d)
                row.append(r)
                col.append(c)

        return COOMatrix(data, row, col, (sr, sc))

    def _to_csc(self) -> 'CSCMatrix':
        return self