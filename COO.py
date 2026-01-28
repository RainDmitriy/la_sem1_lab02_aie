from base import Matrix
from typing import Dict, Tuple, List
from type import COOData, COORows, COOCols, Shape, DenseMatrix

class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.data = list(data)
        self.row = list(row)
        self.col = list(col)
        self.nnz = len(data)
        self._normalize()
        if len(self.data) != len(self.row) or len(self.data) != len(self.col):
            raise ValueError("data, row и col не совпадают после нормализации")
        sr, sc = shape
        for r, c in zip(self.row, self.col):
            if r < 0 or r >= sr or c < 0 or c >= sc:
                raise ValueError(f"индекс ({r},{c}) за границей матрицы {shape}")

    def _to_coo(self) -> 'COOMatrix':
        """COO -> COO (просто возвращаем копию)"""
        return COOMatrix(self.data.copy(), self.row.copy(), self.col.copy(), self.shape)

    def _normalize(self) -> None:
        """Нормализует COO матрицу: суммирует дубликаты, удаляет нули, сортирует"""
        mp = {}
        for d, r, c in zip(self.data, self.row, self.col):
            d_val = float(d)
            if d_val == 0.0:
                continue
            key = (int(r), int(c))
            mp[key] = mp.get(key, 0.0) + d_val
        items = list(mp.items())
        items.sort(key=lambda x: (x[0][0], x[0][1]))

        # Обновляем поля
        self.row = [r for (r, _), _ in items]
        self.col = [c for (_, c), _ in items]
        self.data = [d for _, d in items]
        self.nnz = len(self.data)

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
        """преобразование COOMatrix в CSCMatrix"""
        from CSC import CSCMatrix
        sorted_indices = sorted(range(self.nnz), key=lambda k: (self.col[k], self.row[k]))
        data = []
        indices = []
        indptr = [0] * (self.shape[1] + 1)

        for idx in sorted_indices:
            col = self.col[idx]
            data.append(self.data[idx])
            indices.append(self.row[idx])
            indptr[col + 1] += 1

        for j in range(self.shape[1]):
            indptr[j + 1] += indptr[j]
        return CSCMatrix(data, indices, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """преобразование COOMatrix в CSRMatrix"""
        from CSR import CSRMatrix
        sorted_indices = sorted(range(self.nnz), key=lambda k: (self.row[k], self.col[k]))
        data = []
        indices = []
        indptr = [0] * (self.shape[0] + 1)
        for idx in sorted_indices:
            row = self.row[idx]
            data.append(self.data[idx])
            indices.append(self.col[idx])
            indptr[row + 1] += 1
        for i in range(self.shape[0]):
            indptr[i + 1] += indptr[i]

        return CSRMatrix(data, indices, indptr, self.shape)

    def __str__(self) -> str:
        return f"COOMatrix(shape={self.shape}, nnz={self.nnz})"

    def __repr__(self) -> str:
        return self.__str__()