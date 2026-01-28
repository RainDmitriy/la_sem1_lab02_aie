from CSC import CSCMatrix
from type import Vector
from typing import Tuple, Optional

EPS = 1e-10


def lu_decomposition(mat: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    a = mat.shape[0]
    L = [{} for _ in range(a)]
    U = [{} for _ in range(a)]
    R = [{} for _ in range(a)]
    for j in range(a):
        for p in range(mat.indptr[j], mat.indptr[j + 1]):
            i = mat.indices[p]
            R[i][j] = float(mat.data[p])
    W = [{} for _ in range(a)]
    for i in range(a):
        x = {}
        for j, v in R[i].items():
            if j >= i:
                x[j] = float(v)
        for j, v in W[i].items():
            if j >= i:
                x[j] = x.get(j, 0.0) + float(v)
        p = x.get(i, 0.0)
        if abs(p) < EPS:
            return None
        U[i] = {}
        for j, v in x.items():
            if j >= i and abs(v) > EPS:
                U[i][j] = v
        U[i][i] = p
        L[i][i] = 1.0
        for r in range(i + 1, a):
            z = 0.0
            if i in R[r]:
                z += float(R[r][i])
            if i in W[r]:
                z += float(W[r][i])
            if abs(z) > EPS:
                f = z / p
                L[i][r] = f
                for j, v in U[i].items():
                    if j > i:
                        q = -f * v
                        if abs(q) > EPS:
                            W[r][j] = W[r].get(j, 0.0) + q

    d1, i1, p1 = [], [], [0]
    for j in range(a):
        rr = sorted(L[j].keys())
        for r in rr:
            d1.append(L[j][r])
            i1.append(r)
        p1.append(len(d1))

    C = [{} for _ in range(a)]
    for i in range(a):
        for j, v in U[i].items():
            C[j][i] = v

    d2, i2, p2 = [], [], [0]
    for j in range(a):
        rr = sorted(C[j].keys())
        for r in rr:
            d2.append(C[j][r])
            i2.append(r)
        p2.append(len(d2))

    return (CSCMatrix(d1, i1, p1, (a, a)), CSCMatrix(d2, i2, p2, (a, a)))


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    t = lu_decomposition(A)
    if t is None:
        return None

    L, U = t
    a = A.shape[0]
    l = L.to_dense()
    u = U.to_dense()
    y = [0.0] * a
    for i in range(a):
        s = 0.0
        for j in range(i):
            s += l[i][j] * y[j]
        y[i] = b[i] - s
    x = [0.0] * a
    for i in range(a - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, a):
            s += u[i][j] * x[j]
        if abs(u[i][i]) < EPS:
            return None
        x[i] = (y[i] - s) / u[i][i]
    return x


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    t = lu_decomposition(A)
    if t is None:
        return None

    _, U = t
    a = A.shape[0]
    u = U.to_dense()
    d = 1.0
    for i in range(a):
        d *= u[i][i]
    return d