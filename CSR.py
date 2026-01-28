from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix

# используется для корректной проверки типов
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from CSC import CSCMatrix
    from COO import COOMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)

        # проверки корректности структуры CSR
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
        """Преобразует CSR‑матрицу в плотный формат."""
        sr, sc = self.shape
        dm = [[0.0] * sc for _ in range(sr)]

        for r in range(sr):
            s = self.indptr[r]
            e = self.indptr[r + 1]
            for p in range(s, e):
                c = self.indices[p]
                d = self.data[p]
                dm[r][c] = d

        return dm

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение двух CSR‑матриц через промежуточное COO‑представление."""
        coo_self = self._to_coo()
        coo_other = other._to_coo()
        coo_sum = coo_self._add_impl(coo_other)
        return coo_sum._to_csr()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение матрицы на скаляр."""
        if abs(scalar) < 1e-14:
            # результат – нулевая матрица той же размерности
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)

        data = [float(d) * float(scalar) for d in self.data]
        return CSRMatrix(data, self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование матрицы. Результат возвращается в формате CSC."""
        from CSC import CSCMatrix
        sr, sc = self.shape
        # по условию/подсказке можно просто переинтерпретировать массивы
        return CSCMatrix(self.data[:], self.indices[:], self.indptr[:], (sc, sr))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR‑матрицы на другую матрицу."""
        # приводим вторую матрицу к CSR (если можно)
        if hasattr(other, "_to_csr"):
            b = other._to_csr()
        else:
            from COO import COOMatrix
            b = COOMatrix.from_dense(other.to_dense())._to_csr()

        sr, sc = self.shape
        sr2, sc2 = b.shape
        if sc != sr2:
            raise ValueError("Несовместимые размеры матриц для умножения")

        data = []
        i = []
        indptr = [0] * (sr + 1)

        for r in range(sr):
            mp = {}

            a_s = self.indptr[r]
            a_e = self.indptr[r + 1]

            for ap in range(a_s, a_e):
                k = self.indices[ap]
                da = self.data[ap]

                b_s = b.indptr[k]
                b_e = b.indptr[k + 1]

                for bp in range(b_s, b_e):
                    c = b.indices[bp]
                    db = b.data[bp]

                    mp[c] = mp.get(c, 0.0) + da * db

            # сохраняем ненулевые элементы строки результата
            cols = sorted(mp.keys())
            for c in cols:
                d = mp[c]
                if abs(d) > 1e-14:
                    data.append(d)
                    i.append(c)

            indptr[r + 1] = len(data)

        return CSRMatrix(data, i, indptr, (sr, sc2))

    @classmethod
    def from_dense(cls, dense: DenseMatrix) -> 'CSRMatrix':
        """Создаёт CSR‑матрицу из плотного представления."""
        sr = len(dense)
        sc = len(dense[0]) if sr > 0 else 0

        data = []
        i = []
        indptr = [0]

        for r in range(sr):
            for c in range(sc):
                d = float(dense[r][c])
                if d != 0.0:
                    data.append(d)
                    i.append(c)
            indptr.append(len(data))

        return cls(data, i, indptr, (sr, sc))

    def _to_csc(self) -> 'CSCMatrix':
        """Конвертирует CSR‑матрицу в формат CSC."""
        from CSC import CSCMatrix
        sr, sc = self.shape

        # подсчёт количества элементов в каждом столбце
        cnt = [0] * sc
        for c in self.indices:
            cnt[c] += 1

        indptr = [0] * (sc + 1)
        for c in range(sc):
            indptr[c + 1] = indptr[c] + cnt[c]

        data = [0.0] * len(self.data)
        i = [0] * len(self.indices)
        pos = indptr.copy()

        for r in range(sr):
            s = self.indptr[r]
            e = self.indptr[r + 1]
            for p in range(s, e):
                c = self.indices[p]
                d = self.data[p]

                p2 = pos[c]
                data[p2] = d
                i[p2] = r
                pos[c] += 1

        return CSCMatrix(data, i, indptr, (sr, sc))

    def _to_coo(self) -> 'COOMatrix':
        """Конвертирует CSR‑матрицу в формат COO."""
        from COO import COOMatrix
        sr, sc = self.shape

        data = []
        row = []
        col = []

        for r in range(sr):
            s = self.indptr[r]
            e = self.indptr[r + 1]
            for p in range(s, e):
                c = self.indices[p]
                d = self.data[p]
                data.append(d)
                row.append(r)
                col.append(c)

        return COOMatrix(data, row, col, (sr, sc))

    def _to_csr(self) -> 'CSRMatrix':
        return self