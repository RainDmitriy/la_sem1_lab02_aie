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

    csr = A._to_csr()

    rows: list[dict[int, float]] = []
    col_rows: list[set[int]] = [set() for _ in range(n)]

    for i in range(n):
        d: dict[int, float] = {}
        for p in range(csr.indptr[i], csr.indptr[i + 1]):
            j = csr.indices[p]
            v = float(csr.data[p])
            if abs(v) > _EPS:
                d[j] = d.get(j, 0.0) + v

        for j in list(d.keys()):
            if abs(d[j]) < _EPS:
                del d[j]
            else:
                col_rows[j].add(i)

        rows.append(d)

    rhs = [float(x) for x in b]

    for i in range(n):
        cand = [r for r in col_rows[i] if r >= i]
        if not cand:
            return None

        pivot_row = cand[0]
        best = abs(rows[pivot_row].get(i, 0.0))
        for r in cand[1:]:
            v = abs(rows[r].get(i, 0.0))
            if v > best:
                best = v
                pivot_row = r

        if best < _EPS:
            return None

        if pivot_row != i:
            row_i = rows[i]
            row_p = rows[pivot_row]

            for j in row_i.keys():
                col_rows[j].discard(i)
            for j in row_p.keys():
                col_rows[j].discard(pivot_row)

            rows[i], rows[pivot_row] = rows[pivot_row], rows[i]
            rhs[i], rhs[pivot_row] = rhs[pivot_row], rhs[i]

            row_i = rows[i]
            row_p = rows[pivot_row]
            for j in row_i.keys():
                col_rows[j].add(i)
            for j in row_p.keys():
                col_rows[j].add(pivot_row)

        pivot = rows[i].get(i, 0.0)
        if abs(pivot) < _EPS:
            return None

        pivot_items = list(rows[i].items())

        elim = [r for r in col_rows[i] if r > i]
        for r in elim:
            a_ri = rows[r].get(i, 0.0)
            if abs(a_ri) < _EPS:
                continue
            factor = a_ri / pivot

            if i in rows[r]:
                del rows[r][i]
                col_rows[i].discard(r)

            rr = rows[r]
            for j, a_ij in pivot_items:
                if j <= i:
                    continue
                newv = rr.get(j, 0.0) - factor * a_ij

                if abs(newv) < _EPS:
                    if j in rr:
                        del rr[j]
                        col_rows[j].discard(r)
                else:
                    if j not in rr:
                        col_rows[j].add(r)
                    rr[j] = newv

            rhs[r] -= factor * rhs[i]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        row = rows[i]
        pivot = row.get(i, 0.0)
        if abs(pivot) < _EPS:
            return None
        s = rhs[i]
        for j, v in row.items():
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

    csr = A._to_csr()

    rows: list[dict[int, float]] = []
    col_rows: list[set[int]] = [set() for _ in range(n)]

    for i in range(n):
        d: dict[int, float] = {}
        for p in range(csr.indptr[i], csr.indptr[i + 1]):
            j = csr.indices[p]
            v = float(csr.data[p])
            if abs(v) > _EPS:
                d[j] = d.get(j, 0.0) + v

        for j in list(d.keys()):
            if abs(d[j]) < _EPS:
                del d[j]
            else:
                col_rows[j].add(i)

        rows.append(d)

    sign = 1.0
    det = 1.0

    for i in range(n):
        cand = [r for r in col_rows[i] if r >= i]
        if not cand:
            return 0.0

        pivot_row = cand[0]
        best = abs(rows[pivot_row].get(i, 0.0))
        for r in cand[1:]:
            v = abs(rows[r].get(i, 0.0))
            if v > best:
                best = v
                pivot_row = r

        if best < _EPS:
            return 0.0

        if pivot_row != i:
            row_i = rows[i]
            row_p = rows[pivot_row]

            for j in row_i.keys():
                col_rows[j].discard(i)
            for j in row_p.keys():
                col_rows[j].discard(pivot_row)

            rows[i], rows[pivot_row] = rows[pivot_row], rows[i]
            sign = -sign

            row_i = rows[i]
            row_p = rows[pivot_row]
            for j in row_i.keys():
                col_rows[j].add(i)
            for j in row_p.keys():
                col_rows[j].add(pivot_row)

        pivot = rows[i].get(i, 0.0)
        if abs(pivot) < _EPS:
            return 0.0

        det *= pivot
        pivot_items = list(rows[i].items())

        elim = [r for r in col_rows[i] if r > i]
        for r in elim:
            a_ri = rows[r].get(i, 0.0)
            if abs(a_ri) < _EPS:
                continue
            factor = a_ri / pivot

            if i in rows[r]:
                del rows[r][i]
                col_rows[i].discard(r)

            rr = rows[r]
            for j, a_ij in pivot_items:
                if j <= i:
                    continue
                newv = rr.get(j, 0.0) - factor * a_ij
                if abs(newv) < _EPS:
                    if j in rr:
                        del rr[j]
                        col_rows[j].discard(r)
                else:
                    if j not in rr:
                        col_rows[j].add(r)
                    rr[j] = newv

    return sign * det