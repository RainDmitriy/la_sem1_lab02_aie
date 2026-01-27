from CSC import CSCMatrix
from type import Vector
from typing import Tuple, Optional


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    n = A.shape[0]
    if n != A.shape[1]:
        raise ValueError("LU-разложение применимо только к квадратным матрицам")

    dense = A.to_dense()

    for i in range(n):
        pivot = dense[i][i]
        if abs(pivot) < 1e-10:
            return None

        for k in range(i + 1, n):
            factor = dense[k][i] / pivot
            dense[k][i] = factor
            for j in range(i + 1, n):
                dense[k][j] -= factor * dense[i][j]

    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        L[i][i] = 1.0
        for j in range(n):
            if j >= i:
                U[i][j] = dense[i][j]
            if j < i:
                L[i][j] = dense[i][j]

    L_csc = CSCMatrix.from_dense(L)
    U_csc = CSCMatrix.from_dense(U)

    return (L_csc, U_csc)


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None

    L, U = lu_result
    n = len(b)

    dense_L = L.to_dense()
    dense_U = U.to_dense()

    y = [0.0] * n
    for i in range(n):
        sum_val = b[i]
        for j in range(i):
            sum_val -= dense_L[i][j] * y[j]
        y[i] = sum_val

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_val = y[i]
        for j in range(i + 1, n):
            sum_val -= dense_U[i][j] * x[j]

        if abs(dense_U[i][i]) < 1e-10:
            return None

        x[i] = sum_val / dense_U[i][i]

    return x


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None

    _, U = lu_result
    dense_U = U.to_dense()

    det = 1.0
    n = A.shape[0]

    for i in range(n):
        det *= dense_U[i][i]

    return det