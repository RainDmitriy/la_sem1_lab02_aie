from base import Matrix
from typing import Dict, Tuple, List
from type import COOData, COORows, COOCols, Shape, DenseMatrix

class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        if len(data) != len(row) or len(data) != len(col):
            raise ValueError("data, row и col не совпадают")
        sr, sc = shape
        for r in row:
            if r < 0 or r >= sr:
                raise ValueError(f"row индекс {r} вне диапазона [0, {sr - 1}]")
        for c in col:
            if c < 0 or c >= sc:
                raise ValueError(f"col индекс {c} вне диапазона [0, {sc - 1}]")

        self.data = list(data)
        self.row = list(row)
        self.col = list(col)
        self.nnz = len(data)
        self._normalize()

    def _to_coo(self) -> 'COOMatrix':
        """COO -> COO (просто возвращаем копию)"""
        return COOMatrix(self.data.copy(), self.row.copy(), self.col.copy(), self.shape)

    def _normalize(self) -> None:
        """Нормализует COO матрицу: суммирует дубликаты, удаляет нули, сортирует"""
        mp = {}
        for d, r, c in zip(self.data, self.row, self.col):
            d_val = float(d)
            if abs(d_val) < 1e-14:  # Пропускаем почти нули
                continue
            key = (int(r), int(c))
            mp[key] = mp.get(key, 0.0) + d_val
        items = []
        for (r, c), d in mp.items():
            if abs(d) > 1e-14:
                items.append(((r, c), d))
        items.sort(key=lambda x: (x[0][0], x[0][1]))
        self.row = [r for (r, _), _ in items]
        self.col = [c for (_, c), _ in items]
        self.data = [d for _, d in items]
        self.nnz = len(self.data)  # ОБНОВИТЬ!

    def to_dense(self) -> DenseMatrix:
        """из COO в плотную матрицу"""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]
        for value, r, c in zip(self.data, self.row, self.col):
            dense[r][c] = value
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Реализация сложения двух COO‑матриц."""
        if hasattr(other, "_to_coo"):
            b = other._to_coo()
        else:
            b = COOMatrix.from_dense(other.to_dense())
        mp: Dict[Tuple[int, int], float] = {}
        for d, r, c in zip(self.data, self.row, self.col):
            mp[(r, c)] = mp.get((r, c), 0.0) + float(d)
        for d, r, c in zip(b.data, b.row, b.col):
            mp[(r, c)] = mp.get((r, c), 0.0) + float(d)
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
        data, row, col = [], [], []
        for (r, c), d in mp.items():
            if abs(d) > 1e-14:
                data.append(d)
                row.append(r)
                col.append(c)
        return COOMatrix(data, row, col, (sr, sc))

    @classmethod
    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """создание coo из плотной матрицы"""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        data = []
        row_indices = []
        col_indices = []
        for i in range(rows):
            for j in range(cols):
                value = dense_matrix[i][j]
                if value != 0.0:
                    data.append(float(value))
                    row_indices.append(i)
                    col_indices.append(j)

        return cls(data, row_indices, col_indices, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """Конвертирует текущую матрицу в формат CSC."""
        from CSC import CSCMatrix
        sr, sc = self.shape
        it = list(zip(self.col, self.row, self.data))
        it.sort()
        data = []
        i = []
        indptr = [0] * (sc + 1)
        for c, r, d in it:
            data.append(float(d))
            i.append(int(r))
            indptr[c + 1] += 1
        for c in range(sc):
            indptr[c + 1] += indptr[c]
        return CSCMatrix(data, i, indptr, (sr, sc))

    def _to_csr(self) -> 'CSRMatrix':
        """Конвертирует текущую матрицу в формат CSR."""
        from CSR import CSRMatrix
        sr, sc = self.shape
        it = list(zip(self.row, self.col, self.data))
        it.sort()
        data = []
        i = []
        indptr = [0] * (sr + 1)
        for r, c, d in it:
            data.append(float(d))
            i.append(int(c))
            indptr[r + 1] += 1
        for r in range(sr):
            indptr[r + 1] += indptr[r]
        return CSRMatrix(data, i, indptr, (sr, sc))

    def __str__(self) -> str:
        return f"COOMatrix(shape={self.shape}, nnz={self.nnz})"

    def __repr__(self) -> str:
        return self.__str__()