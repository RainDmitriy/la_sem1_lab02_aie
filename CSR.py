from __future__ import annotations
from typing import List, Dict, TYPE_CHECKING
from base import Matrix
from types import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix
if TYPE_CHECKING:
    from CSC import CSCMatrix
    from COO import COOMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self._validate_inputs(data, indices, indptr, shape)
        self.data: CSRData = list(data)
        self.indices: CSRIndices = list(indices)
        self.indptr: CSRIndptr = list(indptr)

    @staticmethod
    def _validate_inputs(data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape) -> None:
        if len(shape) != 2:
            raise ValueError("shape должен быть кортежем rows, cols")
        n_rows, n_cols = shape
        if n_rows < 0 or n_cols < 0:
            raise ValueError("Размерности матрицы не могут быть отрицательными")
        if len(data) != len(indices):
            raise ValueError("Длины data и indices должны совпадать")
        if len(indptr) != n_rows + 1:
            raise ValueError("indptr должен иметь длину rows + 1")
        if indptr[0] != 0:
            raise ValueError("indptr[0] должен быть 0")
        if indptr[-1] != len(data):
            raise ValueError("indptr[-1] должен быть равен len(data)")
        for k in range(len(indptr) - 1):
            if indptr[k] > indptr[k + 1]:
                raise ValueError("indptr должен быть неубывающим")
        for c in indices:
            if c < 0:
                raise ValueError("indices не могут быть отрицательными")
            if c >= n_cols:
                raise ValueError("indices выходят за границы shape")

    def to_dense(self) -> DenseMatrix:
        n_rows, n_cols = self.shape
        dense: DenseMatrix = [[0.0 for _ in range(n_cols)] for _ in range(n_rows)]
        for r in range(n_rows):
            start = self.indptr[r]
            end = self.indptr[r + 1]
            for p in range(start, end):
                c = int(self.indices[p])
                dense[r][c] += float(self.data[p])
        return dense

    def _add_impl(self, other: Matrix) -> Matrix:
        n_rows, n_cols = self.shape
        other_csr = other if isinstance(other, CSRMatrix) else getattr(other, "_to_csr", None)
        if callable(other_csr):
            other_csr = other_csr()
        if not isinstance(other_csr, CSRMatrix):
            raise TypeError("Нельзя привести вторую матрицу к CSR")
        data: List[float] = []
        indices: List[int] = []
        indptr: List[int] = [0]
        for r in range(n_rows):
            acc: Dict[int, float] = {}
            a_start = self.indptr[r]
            a_end = self.indptr[r + 1]
            for p in range(a_start, a_end):
                c = int(self.indices[p])
                acc[c] = acc.get(c, 0.0) + float(self.data[p])
            b_start = other_csr.indptr[r]
            b_end = other_csr.indptr[r + 1]
            for p in range(b_start, b_end):
                c = int(other_csr.indices[p])
                acc[c] = acc.get(c, 0.0) + float(other_csr.data[p])
            items = [(c, v) for c, v in acc.items() if v != 0.0]
            items.sort(key=lambda t: t[0])
            for c, v in items:
                indices.append(int(c))
                data.append(float(v))
            indptr.append(len(data))
        return CSRMatrix(data, indices, indptr, (n_rows, n_cols))

    def _mul_impl(self, scalar: float) -> Matrix:
        s = float(scalar)
        if s == 0.0:
            n_rows, n_cols = self.shape
            return CSRMatrix([], [], [0] * (n_rows + 1), (n_rows, n_cols))
        data = [float(v) * s for v in self.data]
        return CSRMatrix(data, list(self.indices), list(self.indptr), self.shape)

    def transpose(self) -> Matrix:
        from CSC import CSCMatrix
        n_rows, n_cols = self.shape
        return CSCMatrix(list(self.data), list(self.indices), list(self.indptr), (n_cols, n_rows))

    def _matmul_impl(self, other: Matrix) -> Matrix:
        n_rows, n_mid = self.shape
        n_mid2, n_cols = other.shape
        if n_mid != n_mid2:
            raise ValueError("Несовместимые размерности для умножения")
        right = other if isinstance(other, CSRMatrix) else getattr(other, "_to_csr", None)
        if callable(right):
            right = right()
        if not isinstance(right, CSRMatrix):
            raise TypeError("Нельзя привести правую матрицу к CSR")
        res_data: List[float] = []
        res_indices: List[int] = []
        res_indptr: List[int] = [0]
        for i in range(n_rows):
            acc: Dict[int, float] = {}
            a_start = self.indptr[i]
            a_end = self.indptr[i + 1]
            for ap in range(a_start, a_end):
                k = int(self.indices[ap])
                a_ik = float(self.data[ap])
                if a_ik == 0.0:
                    continue
                b_start = right.indptr[k]
                b_end = right.indptr[k + 1]
                for bp in range(b_start, b_end):
                    j = int(right.indices[bp])
                    b_kj = float(right.data[bp])
                    if b_kj == 0.0:
                        continue
                    acc[j] = acc.get(j, 0.0) + a_ik * b_kj
            items = [(j, v) for j, v in acc.items() if v != 0.0]
            items.sort(key=lambda t: t[0])
            for j, v in items:
                res_indices.append(int(j))
                res_data.append(float(v))
            res_indptr.append(len(res_data))
        return CSRMatrix(res_data, res_indices, res_indptr, (n_rows, n_cols))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> "CSRMatrix":
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
        for r in range(n_rows):
            for c in range(n_cols):
                v = float(dense_matrix[r][c])
                if v != 0.0:
                    data.append(v)
                    indices.append(c)
            indptr.append(len(data))
        return cls(data, indices, indptr, (n_rows, n_cols))

    def _to_csc(self) -> "CSCMatrix":
        from CSC import CSCMatrix
        n_rows, n_cols = self.shape
        col_maps: Dict[int, Dict[int, float]] = {}
        for r in range(n_rows):
            start = self.indptr[r]
            end = self.indptr[r + 1]
            for p in range(start, end):
                c = int(self.indices[p])
                v = float(self.data[p])
                if v == 0.0:
                    continue
                m = col_maps.get(c)
                if m is None:
                    m = {}
                    col_maps[c] = m
                m[r] = m.get(r, 0.0) + v
                if m[r] == 0.0:
                    del m[r]
                    if not m:
                        del col_maps[c]
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

    def _to_coo(self) -> "COOMatrix":
        from COO import COOMatrix
        n_rows, n_cols = self.shape
        data: List[float] = []
        rows: List[int] = []
        cols: List[int] = []
        for r in range(n_rows):
            start = self.indptr[r]
            end = self.indptr[r + 1]
            for p in range(start, end):
                rows.append(r)
                cols.append(int(self.indices[p]))
                data.append(float(self.data[p]))
        return COOMatrix(data, rows, cols, (n_rows, n_cols))