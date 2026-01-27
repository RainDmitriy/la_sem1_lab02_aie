from __future__ import annotations
from typing import List, Dict, TYPE_CHECKING
from base import Matrix
from types import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix
if TYPE_CHECKING:
    from CSR import CSRMatrix
    from COO import COOMatrix

class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self._validate_inputs(data, indices, indptr, shape)
        self.data: CSCData = list(data)
        self.indices: CSCIndices = list(indices)
        self.indptr: CSCIndptr = list(indptr)

    @staticmethod
    def _validate_inputs(data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape) -> None:
        if len(shape) != 2:
            raise ValueError("shape должен быть кортежем rows, cols")
        n_rows, n_cols = shape
        if n_rows < 0 or n_cols < 0:
            raise ValueError("Размерности матрицы не могут быть отрицательными")
        if len(data) != len(indices):
            raise ValueError("Длины data и indices должны совпадать")
        if len(indptr) != n_cols + 1:
            raise ValueError("indptr должен иметь длину cols + 1")
        if indptr[0] != 0:
            raise ValueError("indptr[0] должен быть 0")
        if indptr[-1] != len(data):
            raise ValueError("indptr[-1] должен быть равен len(data)")
        for k in range(len(indptr) - 1):
            if indptr[k] > indptr[k + 1]:
                raise ValueError("indptr должен быть неубывающим")
        for r in indices:
            if r < 0:
                raise ValueError("indices не могут быть отрицательными")
            if r >= n_rows:
                raise ValueError("indices выходят за границы shape")

    def to_dense(self) -> DenseMatrix:
        n_rows, n_cols = self.shape
        dense: DenseMatrix = [[0.0 for _ in range(n_cols)] for _ in range(n_rows)]
        for c in range(n_cols):
            start = self.indptr[c]
            end = self.indptr[c + 1]
            for p in range(start, end):
                r = int(self.indices[p])
                dense[r][c] += float(self.data[p])
        return dense

    def _add_impl(self, other: Matrix) -> Matrix:
        from COO import COOMatrix
        left: COOMatrix = self._to_coo()
        res = left + other
        to_csc = getattr(res, "_to_csc", None)
        if callable(to_csc):
            return to_csc()
        return CSCMatrix.from_dense(res.to_dense())

    def _mul_impl(self, scalar: float) -> Matrix:
        s = float(scalar)
        if s == 0.0:
            n_rows, n_cols = self.shape
            return CSCMatrix([], [], [0] * (n_cols + 1), (n_rows, n_cols))
        data = [float(v) * s for v in self.data]
        return CSCMatrix(data, list(self.indices), list(self.indptr), self.shape)

    def transpose(self) -> Matrix:
        from CSR import CSRMatrix
        n_rows, n_cols = self.shape
        return CSRMatrix(list(self.data), list(self.indices), list(self.indptr), (n_cols, n_rows))
    def _matmul_impl(self, other: Matrix) -> Matrix:
        from CSR import CSRMatrix
        left_csr: CSRMatrix = self._to_csr()
        res = left_csr @ other
        to_csc = getattr(res, "_to_csc", None)
        if callable(to_csc):
            return to_csc()
        raise TypeError("Результат умножения нельзя привести к CSC без dense")

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> "CSCMatrix":
        if not dense_matrix:
            return cls([], [], [0], (0, 0))
        n_rows = len(dense_matrix)
        n_cols = len(dense_matrix[0])
        for r in range(n_rows):
            if len(dense_matrix[r]) != n_cols:
                raise ValueError("Матрица должна быть прямоугольной")
        data: List[float] = []
        indices: List[int] = []
        indptr: List[int] = [0]
        for c in range(n_cols):
            for r in range(n_rows):
                v = float(dense_matrix[r][c])
                if v != 0.0:
                    data.append(v)
                    indices.append(r)
            indptr.append(len(data))
        return cls(data, indices, indptr, (n_rows, n_cols))

    def _to_csr(self) -> "CSRMatrix":
        from CSR import CSRMatrix
        n_rows, n_cols = self.shape
        row_maps: Dict[int, Dict[int, float]] = {}
        for c in range(n_cols):
            start = self.indptr[c]
            end = self.indptr[c + 1]
            for p in range(start, end):
                r = int(self.indices[p])
                v = float(self.data[p])
                if v == 0.0:
                    continue
                m = row_maps.get(r)
                if m is None:
                    m = {}
                    row_maps[r] = m
                m[c] = m.get(c, 0.0) + v
                if m[c] == 0.0:
                    del m[c]
                    if not m:
                        del row_maps[r]
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

    def _to_coo(self) -> "COOMatrix":
        from COO import COOMatrix
        n_rows, n_cols = self.shape
        data: List[float] = []
        rows: List[int] = []
        cols: List[int] = []
        for c in range(n_cols):
            start = self.indptr[c]
            end = self.indptr[c + 1]
            for p in range(start, end):
                rows.append(int(self.indices[p]))
                cols.append(c)
                data.append(float(self.data[p]))
        return COOMatrix(data, rows, cols, (n_rows, n_cols))
