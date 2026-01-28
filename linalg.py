from CSC import CSCMatrix
from CSR import CSRMatrix
from type import Vector
from typing import Tuple, Optional

_EPS = 1e-12 

def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    n_rows, n_cols = A.shape
    if n_rows != n_cols:
        return None
    n = n_rows
    if n == 0:
        empty = CSCMatrix.from_dense([])
        return empty, empty

    a = A.to_dense()

    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = 1.0

    for i in range(n):
        for j in range(i, n):
            s = 0.0
            for k in range(i):
                s += L[i][k] * U[k][j]
            U[i][j] = float(a[i][j]) - s

        pivot = U[i][i]
        if abs(pivot) < _EPS:
            return None

        for j in range(i + 1, n):
            s = 0.0
            for k in range(i):
                s += L[j][k] * U[k][i]
            L[j][i] = (float(a[j][i]) - s) / pivot

    return CSCMatrix.from_dense(L), CSCMatrix.from_dense(U)



def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    n_rows, n_cols = A.shape
    if n_rows != n_cols:
        return None
    n = n_rows
    if len(b) != n:
        return None
    if n == 0:
        return []

    rows: list[dict[int, float]] = [dict() for _ in range(n)]
    _, n_colsA = A.shape
    for j in range(n_colsA):
        start, end = A.indptr[j], A.indptr[j + 1]
        for p in range(start, end):
            i = A.indices[p]
            v = float(A.data[p])
            if v != 0.0:
                rows[i][j] = rows[i].get(j, 0.0) + v

    rhs = [float(bi) for bi in b]

    for i in range(n):
        pivot = rows[i].get(i, 0.0)
        if abs(pivot) < _EPS:
            return None

        for r in range(i + 1, n):
            a_ri = rows[r].get(i, 0.0)
            if abs(a_ri) < _EPS:
                continue

            factor = a_ri / pivot
            rows[r][i] = factor

            row_i = rows[i]
            row_r = rows[r]

            for j, u_ij in row_i.items():
                if j <= i:
                    continue
                newv = row_r.get(j, 0.0) - factor * u_ij
                if abs(newv) < _EPS:
                    row_r.pop(j, None)
                else:
                    row_r[j] = newv

            rhs[r] -= factor * rhs[i]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        row_i = rows[i]
        pivot = row_i.get(i, 0.0)
        if abs(pivot) < _EPS:
            return None

        s = rhs[i]
        for j, v in row_i.items():
            if j > i:
                s -= v * x[j]
        x[i] = s / pivot

    return x


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    n_rows, n_cols = A.shape
    if n_rows != n_cols:
        return None
    n = n_rows
    if n == 0:
        return 1.0

    rows: list[dict[int, float]] = [dict() for _ in range(n)]
    _, n_colsA = A.shape
    for j in range(n_colsA):
        start, end = A.indptr[j], A.indptr[j + 1]
        for p in range(start, end):
            i = A.indices[p]
            v = float(A.data[p])
            if v != 0.0:
                rows[i][j] = rows[i].get(j, 0.0) + v

    for i in range(n):
        pivot = rows[i].get(i, 0.0)
        if abs(pivot) < _EPS:
            return None

        for r in range(i + 1, n):
            a_ri = rows[r].get(i, 0.0)
            if abs(a_ri) < _EPS:
                continue

            factor = a_ri / pivot
            rows[r][i] = factor  

            row_i = rows[i]
            row_r = rows[r]
            for j, u_ij in row_i.items():
                if j <= i:
                    continue
                newv = row_r.get(j, 0.0) - factor * u_ij
                if abs(newv) < _EPS:
                    row_r.pop(j, None)
                else:
                    row_r[j] = newv

    det = 1.0
    for i in range(n):
        det *= rows[i].get(i, 0.0)
    return det
