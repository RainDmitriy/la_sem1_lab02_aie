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
    if n != m:
        return None
    if len(b) != n:
        return None
    if n == 0:
        return []

    csr = A._to_csr()

    def matvec(x: list[float]) -> list[float]:
        y = [0.0] * n
        data = csr.data
        idx = csr.indices
        ip = csr.indptr
        for i in range(n):
            s = 0.0
            for p in range(ip[i], ip[i + 1]):
                s += float(data[p]) * x[idx[p]]
            y[i] = s
        return y

    diag = [0.0] * n
    data = csr.data
    idx = csr.indices
    ip = csr.indptr
    for i in range(n):
        di = 0.0
        for p in range(ip[i], ip[i + 1]):
            if idx[p] == i:
                di += float(data[p])
        diag[i] = di

    def Minv(v: list[float]) -> list[float]:
        out = [0.0] * n
        for i in range(n):
            d = diag[i]
            if abs(d) > _EPS:
                out[i] = v[i] / d
            else:
                out[i] = v[i]
        return out

    bvec = [float(x) for x in b]
    x = [0.0] * n

    r = [bvec[i] for i in range(n)] 
    r_hat = [ri for ri in r]

    rho_old = 1.0
    alpha = 1.0
    omega = 1.0

    v = [0.0] * n
    p = [0.0] * n

    b_norm2 = 0.0
    for bi in bvec:
        b_norm2 += bi * bi
    if b_norm2 < _EPS:
        return [0.0] * n

    tol = 1e-9 
    tol2 = (tol * tol) * b_norm2

    max_iter = min(2000, max(50, 5 * n)) 

    for _ in range(max_iter):
        r_norm2 = 0.0
        for ri in r:
            r_norm2 += ri * ri
        if r_norm2 <= tol2:
            return x

        rho_new = 0.0
        for i in range(n):
            rho_new += r_hat[i] * r[i]
        if abs(rho_new) < _EPS:
            break

        beta = (rho_new / rho_old) * (alpha / omega)
        rho_old = rho_new

        for i in range(n):
            p[i] = r[i] + beta * (p[i] - omega * v[i])

        phat = Minv(p)
        v = matvec(phat)

        rhat_v = 0.0
        for i in range(n):
            rhat_v += r_hat[i] * v[i]
        if abs(rhat_v) < _EPS:
            break

        alpha = rho_new / rhat_v

        s = [0.0] * n
        for i in range(n):
            s[i] = r[i] - alpha * v[i]

        s_norm2 = 0.0
        for si in s:
            s_norm2 += si * si
        if s_norm2 <= tol2:
            for i in range(n):
                x[i] += alpha * phat[i]
            return x

        shat = Minv(s)
        t = matvec(shat)

        t_dot_s = 0.0
        t_dot_t = 0.0
        for i in range(n):
            t_dot_s += t[i] * s[i]
            t_dot_t += t[i] * t[i]
        if abs(t_dot_t) < _EPS:
            break

        omega = t_dot_s / t_dot_t
        if abs(omega) < _EPS:
            break

        for i in range(n):
            x[i] += alpha * phat[i] + omega * shat[i]
            r[i] = s[i] - omega * t[i]

    return None

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