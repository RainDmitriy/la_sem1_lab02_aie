from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix
from typing import Dict, Tuple, List


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
        self.shape = shape

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу"""
        sr, sc = self.shape
        dm = [[0.0 for _ in range(sc)] for _ in range(sr)]
        for d, r, c in zip(self.data, self.row, self.col):
            dm[r][c] += float(d)
        return dm

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Реализация сложения двух COO‑матриц."""
        # приводим вторую матрицу к COO
        if hasattr(other, "_to_coo"):
            b = other._to_coo()
        else:
            b = COOMatrix.from_dense(other.to_dense())

        mp: Dict[Tuple[int, int], float] = {}
        # добавляем элементы первой матрицы
        for d, r, c in zip(self.data, self.row, self.col):
            mp[(r, c)] = mp.get((r, c), 0.0) + float(d)
        # добавляем элементы второй матрицы
        for d, r, c in zip(b.data, b.row, b.col):
            mp[(r, c)] = mp.get((r, c), 0.0) + float(d)

        # собираем только ненулевые элементы
        data, row, col = [], [], []
        for (r, c), d in mp.items():
            if abs(d) > 1e-14:
                data.append(d)
                row.append(r)
                col.append(c)
        return COOMatrix(data, row, col, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение матрицы на число."""
        data = [float(d) * float(scalar) for d in self.data]
        return COOMatrix(data, self.row[:], self.col[:], self.shape)

    def transpose(self) -> 'Matrix':
        """Возвращает транспонированную матрицу."""
        sr, sc = self.shape
        return COOMatrix(self.data[:], self.col[:], self.row[:], (sc, sr))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Реализация умножения матриц."""
        if self.shape[1] != other.shape[0]:
            raise ValueError("Несовместимые размеры матриц для умножения")
        sr, sc = self.shape[0], other.shape[1]
        mp = {}
        # конвертируем правую матрицу в CSR для эффективного доступа
        if hasattr(other, "_to_csr"):
            b = other._to_csr()
        else:
            b = COOMatrix.from_dense(other.to_dense())._to_csr()

        for p in range(len(self.data)):
            r = self.row[p]
            c = self.col[p]
            da = float(self.data[p])
            s = b.indptr[c]
            e = b.indptr[c + 1]
            for p2 in range(s, e):
                c2 = b.indices[p2]
                db = float(b.data[p2])
                key = (r, c2)
                mp[key] = mp.get(key, 0.0) + da * db

        # преобразуем результат обратно в COO
        data, row, col = [], [], []
        for (r, c), d in mp.items():
            if abs(d) > 1e-14:
                data.append(d)
                row.append(r)
                col.append(c)
        return COOMatrix(data, row, col, (sr, sc))

    @classmethod
    def from_dense(cls, dense: DenseMatrix) -> 'COOMatrix':
        """Создаёт COO‑матрицу из плотного представления."""
        data, row, col = [], [], []
        sr = len(dense)
        sc = len(dense[0]) if sr > 0 else 0
        for r in range(sr):
            for c in range(sc):
                d = float(dense[r][c])
                if d != 0.0:
                    data.append(d)
                    row.append(r)
                    col.append(c)
        return cls(data, row, col, (sr, sc))

    def _to_csc(self) -> 'CSCMatrix':
        """Конвертирует текущую матрицу в формат CSC."""
        from CSC import CSCMatrix
        sr, sc = self.shape
        # сортируем элементы по столбцам, затем по строкам
        it = list(zip(self.col, self.row, self.data))
        it.sort()
        data = []
        i = []
        indptr = [0] * (sc + 1)
        for c, r, d in it:
            data.append(float(d))
            i.append(int(r))
            indptr[c + 1] += 1
        # аккумулируем указатели столбцов
        for c in range(sc):
            indptr[c + 1] += indptr[c]
        return CSCMatrix(data, i, indptr, (sr, sc))

    def _to_csr(self) -> 'CSRMatrix':
        """Конвертирует текущую матрицу в формат CSR."""
        from CSR import CSRMatrix
        sr, sc = self.shape
        # сортируем элементы по строкам, затем по столбцам
        it = list(zip(self.row, self.col, self.data))
        it.sort()
        data = []
        i = []
        indptr = [0] * (sr + 1)
        for r, c, d in it:
            data.append(float(d))
            i.append(int(c))
            indptr[r + 1] += 1
        # аккумулируем указатели строк
        for r in range(sr):
            indptr[r + 1] += indptr[r]
        return CSRMatrix(data, i, indptr, (sr, sc))

    def _normalize(self) -> None:
        """Нормализует матрицу: удаляет дубликаты и сортирует."""
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

    def _to_coo(self) -> 'COOMatrix':
        """COO -> COO (просто возвращаем копию)"""
        return COOMatrix(self.data[:], self.row[:], self.col[:], self.shape)

    def __str__(self) -> str:
        return f"COOMatrix(shape={self.shape}, nnz={len(self.data)})"

    def __repr__(self) -> str:
        return self.__str__()