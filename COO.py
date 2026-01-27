from __future__ import annotations
from typing import List, Tuple, Dict, TYPE_CHECKING
from base import Matrix
from types import COOData, COORows, COOCols, Shape, DenseMatrix
if TYPE_CHECKING:
    from CSR import CSRMatrix
    from CSC import CSCMatrix


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self._validate_inputs(data, row, col, shape)
        self.data: COOData = list(data)
        self.row: COORows = list(row)
        self.col: COOCols = list(col)

    @staticmethod
    def _validate_inputs(data: COOData, row: COORows, col: COOCols, shape: Shape) -> None:
        if len(shape) != 2:
            raise ValueError("shape должен быть кортежем rows, cols")
        n_rows, n_cols = shape
        if n_rows < 0 or n_cols < 0:
            raise ValueError("Размерности матрицы не могут быть отрицательными")
        if not (len(data) == len(row) == len(col)):
            raise ValueError("Длины data, row, col должны совпадать")
        for r, c in zip(row, col):
            if r < 0 or c < 0:
                raise ValueError("Индексы row col не могут быть отрицательными")
            if r >= n_rows or c >= n_cols:
                raise ValueError("Индексы row col выходят за границы shape")

    def to_dense(self) -> DenseMatrix:
        n_rows, n_cols = self.shape
        dense: DenseMatrix = [[0.0 for _ in range(n_cols)] for _ in range(n_rows)]
        for v, r, c in zip(self.data, self.row, self.col):
            dense[int(r)][int(c)] += float(v)
        return dense

    @staticmethod
    def _as_coo(other: Matrix) -> "COOMatrix":
        if isinstance(other, COOMatrix):
            return other
        to_coo = getattr(other, "_to_coo", None)
        if callable(to_coo):
            return to_coo()
        raise TypeError("Нельзя привести матрицу к COO")

    def _add_impl(self, other: Matrix) -> Matrix:
        other_coo = self._as_coo(other)
        acc: Dict[Tuple[int, int], float] = {}
        for v, r, c in zip(self.data, self.row, self.col):
            key = (int(r), int(c))
            acc[key] = acc.get(key, 0.0) + float(v)
        for v, r, c in zip(other_coo.data, other_coo.row, other_coo.col):
            key = (int(r), int(c))
            acc[key] = acc.get(key, 0.0) + float(v)
        items = [((r, c), v) for (r, c), v in acc.items() if v != 0.0]
        items.sort(key=lambda t: (t[0][0], t[0][1]))
        data: List[float] = []
        rows: List[int] = []
        cols: List[int] = []
        for (r, c), v in items:
            rows.append(r)
            cols.append(c)
            data.append(v)
        return COOMatrix(data, rows, cols, self.shape)

    def _mul_impl(self, scalar: float) -> Matrix:
        s = float(scalar)
        if s == 0.0:
            return COOMatrix([], [], [], self.shape)
        data = [float(v) * s for v in self.data]
        return COOMatrix(data, list(self.row), list(self.col), self.shape)

    def transpose(self) -> Matrix:
        n_rows, n_cols = self.shape
        return COOMatrix(list(self.data), list(self.col), list(self.row), (n_cols, n_rows))

    def _matmul_impl(self, other: Matrix) -> Matrix:
        from CSR import CSRMatrix
        left_csr: CSRMatrix = self._to_csr()
        res = left_csr @ other
        to_coo = getattr(res, "_to_coo", None)
        if callable(to_coo):
            return to_coo()
        return COOMatrix.from_dense(res.to_dense())

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> "COOMatrix":
        if not dense_matrix:
            return cls([], [], [], (0, 0))
        n_rows = len(dense_matrix)
        n_cols = len(dense_matrix[0])
        for r in range(n_rows):
            if len(dense_matrix[r]) != n_cols:
                raise ValueError("Матрица должна быть прямоугольной")
        data: List[float] = []
        rows: List[int] = []
        cols: List[int] = []
        for i in range(n_rows):
            for j in range(n_cols):
                v = float(dense_matrix[i][j])
                if v != 0.0:
                    rows.append(i)
                    cols.append(j)
                    data.append(v)
        return cls(data, rows, cols, (n_rows, n_cols))

    def _to_csc(self) -> "CSCMatrix":
        from CSC import CSCMatrix
        n_rows, n_cols = self.shape
        col_maps: Dict[int, Dict[int, float]] = {}
        for v, r, c in zip(self.data, self.row, self.col):
            rr = int(r)
            cc = int(c)
            val = float(v)
            if val == 0.0:
                continue
            m = col_maps.get(cc)
            if m is None:
                m = {}
                col_maps[cc] = m
            m[rr] = m.get(rr, 0.0) + val
            if m[rr] == 0.0:
                del m[rr]
                if not m:
                    del col_maps[cc]
        data: List[float] = []
        indices: List[int] = []
        indptr: List[int] = [0]
        for c in range(n_cols):
            m = col_maps.get(c)
            if not m:
                indptr.append(len(data))
                continue
            items = [(r, v) for r, v in m.items() if v != 0.0]
            items.sort(key=lambda t: t[0])
            for r, v in items:
                indices.append(int(r))
                data.append(float(v))
            indptr.append(len(data))
        return CSCMatrix(data, indices, indptr, (n_rows, n_cols))

    def _to_csr(self) -> "CSRMatrix":
        from CSR import CSRMatrix
        n_rows, n_cols = self.shape
        row_maps: Dict[int, Dict[int, float]] = {}
        for v, r, c in zip(self.data, self.row, self.col):
            rr = int(r)
            cc = int(c)
            val = float(v)
            if val == 0.0:
                continue
            m = row_maps.get(rr)
            if m is None:
                m = {}
                row_maps[rr] = m
            m[cc] = m.get(cc, 0.0) + val
            if m[cc] == 0.0:
                del m[cc]
                if not m:
                    del row_maps[rr]
        data: List[float] = []
        indices: List[int] = []
        indptr: List[int] = [0]
        for r in range(n_rows):
            m = row_maps.get(r)
            if not m:
                indptr.append(len(data))
                continue
            items = [(c, v) for c, v in m.items() if v != 0.0]
            items.sort(key=lambda t: t[0])
            for c, v in items:
                indices.append(int(c))
                data.append(float(v))
            indptr.append(len(data))
        return CSRMatrix(data, indices, indptr, (n_rows, n_cols))