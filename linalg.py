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
    col_rows: list[set[int]] = [set() for _ in range(n)]

    for j in range(n):
        start, end = A.indptr[j], A.indptr[j + 1]
        for p in range(start, end):
            i = A.indices[p]
            v = float(A.data[p])
            if abs(v) < _EPS:
                continue
            rows[i][j] = rows[i].get(j, 0.0) + v
            col_rows[j].add(i)

    rhs = [float(bi) for bi in b]

    for i in range(n):
        pivot_row = -1
        best = 0.0
        for r in col_rows[i]:
            if r >= i:
                v = abs(rows[r].get(i, 0.0))
                if v > best:
                    best = v
                    pivot_row = r
        if pivot_row == -1 or best < _EPS:
            return None

        if pivot_row != i:
            ri = rows[i]
            rp = rows[pivot_row]
            cols_i = set(ri.keys())
            cols_p = set(rp.keys())

            only_i = cols_i - cols_p
            only_p = cols_p - cols_i

            for c in only_i:
                col_rows[c].discard(i)
                col_rows[c].add(pivot_row)
            for c in only_p:
                col_rows[c].discard(pivot_row)
                col_rows[c].add(i)

            rows[i], rows[pivot_row] = rows[pivot_row], rows[i]
            rhs[i], rhs[pivot_row] = rhs[pivot_row], rhs[i]

        row_i = rows[i]
        pivot = row_i.get(i, 0.0)
        if abs(pivot) < _EPS:
            return None

        rhs_i = rhs[i]
        u_items = [(j, v) for j, v in row_i.items() if j > i]

        affected = [r for r in col_rows[i] if r > i]
        for r in affected:
            row_r = rows[r]
            a_ri = row_r.get(i, 0.0)
            if abs(a_ri) < _EPS:
                col_rows[i].discard(r)
                row_r.pop(i, None)
                continue

            factor = a_ri / pivot

            row_r.pop(i, None)
            col_rows[i].discard(r)

            for j, u_ij in u_items:
                newv = row_r.get(j, 0.0) - factor * u_ij
                if abs(newv) < _EPS:
                    if j in row_r:
                        del row_r[j]
                        col_rows[j].discard(r)
                else:
                    if j not in row_r:
                        col_rows[j].add(r)
                    row_r[j] = newv

            rhs[r] -= factor * rhs_i

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
    col_rows: list[set[int]] = [set() for _ in range(n)]

    for j in range(n):
        start, end = A.indptr[j], A.indptr[j + 1]
        for p in range(start, end):
            i = A.indices[p]
            v = float(A.data[p])
            if abs(v) < _EPS:
                continue
            rows[i][j] = rows[i].get(j, 0.0) + v
            col_rows[j].add(i)

    sign = 1.0

    for i in range(n):
        pivot_row = -1
        best = 0.0
        for r in col_rows[i]:
            if r >= i:
                v = abs(rows[r].get(i, 0.0))
                if v > best:
                    best = v
                    pivot_row = r
        if pivot_row == -1 or best < _EPS:
            return None

        if pivot_row != i:
            ri = rows[i]
            rp = rows[pivot_row]
            cols_i = set(ri.keys())
            cols_p = set(rp.keys())

            only_i = cols_i - cols_p
            only_p = cols_p - cols_i

            for c in only_i:
                col_rows[c].discard(i)
                col_rows[c].add(pivot_row)
            for c in only_p:
                col_rows[c].discard(pivot_row)
                col_rows[c].add(i)

            rows[i], rows[pivot_row] = rows[pivot_row], rows[i]
            sign = -sign

        row_i = rows[i]
        pivot = row_i.get(i, 0.0)
        if abs(pivot) < _EPS:
            return None

        u_items = [(j, v) for j, v in row_i.items() if j > i]

        affected = [r for r in col_rows[i] if r > i]
        for r in affected:
            row_r = rows[r]
            a_ri = row_r.get(i, 0.0)
            if abs(a_ri) < _EPS:
                col_rows[i].discard(r)
                row_r.pop(i, None)
                continue

            factor = a_ri / pivot

            row_r.pop(i, None)
            col_rows[i].discard(r)

            for j, u_ij in u_items:
                newv = row_r.get(j, 0.0) - factor * u_ij
                if abs(newv) < _EPS:
                    if j in row_r:
                        del row_r[j]
                        col_rows[j].discard(r)
                else:
                    if j not in row_r:
                        col_rows[j].add(r)
                    row_r[j] = newv

    det = sign
    for i in range(n):
        det *= rows[i].get(i, 0.0)
    return det