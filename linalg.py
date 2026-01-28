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

    a = A.to_dense()

    for i in range(n):
        for j in range(i, n):
            s = 0.0
            for k in range(i):
                s += a[i][k] * a[k][j] 
            a[i][j] = float(a[i][j]) - s

        pivot = a[i][i]
        if abs(pivot) < _EPS:
            return None

        for j in range(i + 1, n):
            s = 0.0
            for k in range(i):
                s += a[j][k] * a[k][i]
            a[j][i] = (float(a[j][i]) - s) / pivot

    y = [0.0] * n
    for i in range(n):
        s = float(b[i])
        for j in range(i):
            s -= float(a[i][j]) * y[j]
        y[i] = s

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= float(a[i][j]) * x[j]
        pivot = float(a[i][i])
        if abs(pivot) < _EPS:
            return None
        x[i] = s / pivot

    return x


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    n_rows, n_cols = A.shape
    if n_rows != n_cols:
        return None
    n = n_rows
    if n == 0:
        return 1.0

    a = A.to_dense()

    for i in range(n):
        for j in range(i, n):
            s = 0.0
            for k in range(i):
                s += a[i][k] * a[k][j]
            a[i][j] = float(a[i][j]) - s

        pivot = a[i][i]
        if abs(pivot) < _EPS:
            return None

        for j in range(i + 1, n):
            s = 0.0
            for k in range(i):
                s += a[j][k] * a[k][i]
            a[j][i] = (float(a[j][i]) - s) / pivot

    det = 1.0
    for i in range(n):
        det *= float(a[i][i])
    return det

