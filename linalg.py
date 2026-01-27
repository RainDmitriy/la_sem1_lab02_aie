from CSC import CSCMatrix
from type import Vector
from typing import Tuple, Optional


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    n = A.shape[0]
    if n != A.shape[1]:
        raise ValueError("LU-разложение применимо только к квадратным матрицам")

    dense = A.to_dense()

    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for k in range(i, n):
            sum_val = 0.0
            for j in range(i):
                sum_val += L[i][j] * U[j][k]
            U[i][k] = dense[i][k] - sum_val

        L[i][i] = 1.0
        for k in range(i + 1, n):
            sum_val = 0.0
            for j in range(i):
                sum_val += L[k][j] * U[j][i]

            if abs(U[i][i]) < 1e-10:
                return None

            L[k][i] = (dense[k][i] - sum_val) / U[i][i]

    L_csc = CSCMatrix.from_dense(L)
    U_csc = CSCMatrix.from_dense(U)

    return (L_csc, U_csc)


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None

    L, U = lu_result
    n = len(b)

    y = [0.0] * n
    dense_L = L.to_dense()
    for i in range(n):
        sum_val = 0.0
        for j in range(i):
            sum_val += dense_L[i][j] * y[j]
        y[i] = b[i] - sum_val

    x = [0.0] * n
    dense_U = U.to_dense()

    for i in range(n - 1, -1, -1):
        sum_val = 0.0
        for j in range(i + 1, n):
            sum_val += dense_U[i][j] * x[j]

        if abs(dense_U[i][i]) < 1e-10:
            return None

        x[i] = (y[i] - sum_val) / dense_U[i][i]

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