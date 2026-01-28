from CSC import CSCMatrix
from CSR import CSRMatrix
from my_types import Vector
from typing import Tuple, Optional

def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    n, m = A.shape
    if n != m: return None

    a = A.to_dense()
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n): L[i][i] = 1.0

    for k in range(n):
        s = sum(L[k][p] * U[p][k] for p in range(k))
        pivot = a[k][k] - s
        if pivot == 0: return None
        U[k][k] = pivot

        for j in range(k + 1, n):
            s = sum(L[k][p] * U[p][j] for p in range(k))
            U[k][j] = a[k][j] - s

        for i in range(k + 1, n):
            s = sum(L[i][p] * U[p][k] for p in range(k))
            L[i][k] = (a[i][k] - s) / U[k][k]

    return CSCMatrix.from_dense(L), CSCMatrix.from_dense(U)

def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    res = lu_decomposition(A)
    if res is None: return None
    L, U = res

    n, m = A.shape
    if len(b) != n: return None

    Ld, Ud = L.to_dense(), U.to_dense()

    y = [0.0] * n
    for i in range(n):
        s = sum(Ld[i][j] * y[j] for j in range(i))
        y[i] = b[i] - s

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(Ud[i][j] * x[j] for j in range(i + 1, n))
        if Ud[i][i] == 0: return None
        x[i] = (y[i] - s) / Ud[i][i]

    return x

def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    res = lu_decomposition(A)
    if res is None: return None
    _, U = res
    Ud = U.to_dense()
    det = 1.0
    for i in range(len(Ud)):
        det *= Ud[i][i]
    return det