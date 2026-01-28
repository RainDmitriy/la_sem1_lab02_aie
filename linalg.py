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
    if any(len(row) != n for row in a):
        return None
    
    # Инициализация L (единичная), U (нулевая)
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

    lu = lu_decomposition(A)
    if lu is None:
        return None
    L, U = lu

    Ld = L.to_dense()
    Ud = U.to_dense()

    y: list[float] = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += Ld[i][j] * y[j]
        y[i] = float(b[i]) - s

    x: list[float] = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += Ud[i][j] * x[j]
        pivot = Ud[i][i]
        if abs(pivot) < _EPS:
            return None
        x[i] = (y[i] - s) / pivot

    return x


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    n_rows, n_cols = A.shape
    if n_rows != n_cols:
        return None

    n = n_rows
    if n == 0:
        return 1.0 

    lu = lu_decomposition(A)
    if lu is None:
        return None
    _, U = lu

    Ud = U.to_dense()
    det = 1.0
    for i in range(n):
        det *= Ud[i][i]
    return det

