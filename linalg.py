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
    n, m = A.shape
    if n != m or len(b) != n:
        return None
    if n == 0:
        return []

    csr = A._to_csr()

    rows: list[dict[int, float]] = [dict() for _ in range(n)]
    col_rows: list[list[int]] = [[] for _ in range(n)]

    for row_id in range(n):
        d = rows[row_id]
        for p in range(csr.indptr[row_id], csr.indptr[row_id + 1]):
            j = csr.indices[p]
            v = float(csr.data[p])
            if abs(v) <= _EPS:
                continue
            prev = d.get(j, 0.0)
            newv = prev + v
            if abs(newv) <= _EPS:
                if j in d:
                    del d[j]
            else:
                if j not in d:
                    col_rows[j].append(row_id)
                d[j] = newv

    pos_to_row = list(range(n))
    row_to_pos = list(range(n))

    rhs = [float(x) for x in b]

    for i in range(n):
        best_pos = -1
        best_abs = 0.0

        for row_id in col_rows[i]:
            pos = row_to_pos[row_id]
            if pos < i:
                continue
            v = rows[row_id].get(i, 0.0)
            av = abs(v)
            if av > best_abs:
                best_abs = av
                best_pos = pos

        if best_abs <= _EPS:
            return None

        if best_pos != i:
            r1 = pos_to_row[i]
            r2 = pos_to_row[best_pos]
            pos_to_row[i], pos_to_row[best_pos] = r2, r1
            row_to_pos[r1], row_to_pos[r2] = best_pos, i
            rhs[i], rhs[best_pos] = rhs[best_pos], rhs[i]

        pivot_row_id = pos_to_row[i]
        pivot_row = rows[pivot_row_id]
        pivot = pivot_row.get(i, 0.0)
        if abs(pivot) <= _EPS:
            return None

        pivot_items = [(j, v) for j, v in pivot_row.items() if j > i]

        for row_id in col_rows[i]:
            pos = row_to_pos[row_id]
            if pos <= i:
                continue

            rr = rows[row_id]
            a_ri = rr.get(i, 0.0)
            if abs(a_ri) <= _EPS:
                continue

            factor = a_ri / pivot

            del rr[i]

            for j, a_ij in pivot_items:
                newv = rr.get(j, 0.0) - factor * a_ij
                if abs(newv) <= _EPS:
                    if j in rr:
                        del rr[j]
                else:
                    if j not in rr:
                        col_rows[j].append(row_id) 
                    rr[j] = newv

            rhs[pos] -= factor * rhs[i]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        row_id = pos_to_row[i]
        row = rows[row_id]
        piv = row.get(i, 0.0)
        if abs(piv) <= _EPS:
            return None

        s = rhs[i]
        for j, v in row.items():
            if j > i:
                s -= v * x[j]
        x[i] = s / piv

    return x

def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    n, m = A.shape
    if n != m:
        return None
    if n == 0:
        return 1.0

    csr = A._to_csr()

    rows: list[dict[int, float]] = [dict() for _ in range(n)]
    col_list: list[list[int]] = [[] for _ in range(n)]

    for row_id in range(n):
        d = rows[row_id]
        for p in range(csr.indptr[row_id], csr.indptr[row_id + 1]):
            j = csr.indices[p]
            v = float(csr.data[p])
            if abs(v) > _EPS:
                prev = d.get(j, 0.0)
                newv = prev + v
                if abs(newv) < _EPS:
                    if j in d:
                        del d[j]
                else:
                    if j not in d:
                        col_list[j].append(row_id)
                    d[j] = newv

    pos_to_row = list(range(n))
    row_to_pos = list(range(n))

    sign = 1.0
    det = 1.0

    for i in range(n):
        best_pos = -1
        best_abs = 0.0

        for row_id in col_list[i]:
            pos = row_to_pos[row_id]
            if pos < i:
                continue
            v = rows[row_id].get(i, 0.0)
            av = abs(v)
            if av > best_abs:
                best_abs = av
                best_pos = pos

        if best_abs < _EPS:
            return 0.0

        if best_pos != i:
            row_i = pos_to_row[i]
            row_p = pos_to_row[best_pos]

            pos_to_row[i], pos_to_row[best_pos] = row_p, row_i
            row_to_pos[row_i], row_to_pos[row_p] = best_pos, i
            sign = -sign

        pivot_row_id = pos_to_row[i]
        pivot = rows[pivot_row_id].get(i, 0.0)
        if abs(pivot) < _EPS:
            return 0.0

        det *= pivot

        pivot_items = list(rows[pivot_row_id].items())

        for row_id in col_list[i]:
            pos = row_to_pos[row_id]
            if pos <= i:
                continue
            rr = rows[row_id]
            a_ri = rr.get(i, 0.0)
            if abs(a_ri) < _EPS:
                continue
            factor = a_ri / pivot

            del rr[i]

            for j, a_ij in pivot_items:
                if j <= i:
                    continue
                newv = rr.get(j, 0.0) - factor * a_ij
                if abs(newv) < _EPS:
                    if j in rr:
                        del rr[j]
                else:
                    if j not in rr:
                        col_list[j].append(row_id)
                    rr[j] = newv

    return sign * det