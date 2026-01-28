from CSC import CSCMatrix
from matrix_types import Vector
from typing import Tuple, Optional


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("LU-разложение определено только для квадратных матриц")
    n = A.shape[0]
    if A.nnz < n * n * 0.3:
        return _lu_decomposition_sparse(A)
    else:
        dense = A.to_dense()
        L = [[0.0] * n for _ in range(n)]
        U = [[0.0] * n for _ in range(n)]
        for i in range(n):
            L[i][i] = 1.0
        for i in range(n):
            # U[i][j]
            for j in range(i, n):
                s = 0.0
                for k in range(i):
                    s += L[i][k] * U[k][j]
                U[i][j] = dense[i][j] - s
            # L[j][i]
            for j in range(i + 1, n):
                s = 0.0
                for k in range(i):
                    s += L[j][k] * U[k][i]

                if abs(U[i][i]) < 1e-12:
                    return None
                L[j][i] = (dense[j][i] - s) / U[i][i]
        L_csc = CSCMatrix.from_dense(L)
        U_csc = CSCMatrix.from_dense(U)
        return L_csc, U_csc


def _lu_decomposition_sparse(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    Упрощенная версия LU для разреженных матриц
    """
    n = A.shape[0]
    dense = A.to_dense()

    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = 1.0
    for i in range(n):
        for j in range(i, n):
            s = 0.0
            for k in range(i):
                s += L[i][k] * U[k][j]
            U[i][j] = dense[i][j] - s
        for j in range(i + 1, n):
            s = 0.0
            for k in range(i):
                s += L[j][k] * U[k][i]
            if abs(U[i][i]) < 1e-12:
                return None
            L[j][i] = (dense[j][i] - s) / U[i][i]
    L_csc = CSCMatrix.from_dense(L)
    U_csc = CSCMatrix.from_dense(U)
    return L_csc, U_csc


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение
    """
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None
    L, U = lu_result
    n = len(b)
    L_dense = L.to_dense()
    U_dense = U.to_dense()
    y = [0.0] * n
    for i in range(n):
        s = b[i]
        for j in range(i):
            s -= L_dense[i][j] * y[j]
        y[i] = s  # L[i][i] = 1.0
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(U_dense[i][i]) < 1e-12:
            return None
        s = y[i]
        for j in range(i + 1, n):
            s -= U_dense[i][j] * x[j]
        x[i] = s / U_dense[i][i]

    return x


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None
    L, U = lu_result
    n = len(b)
    L_dense = L.to_dense()
    U_dense = U.to_dense()

    y = [0.0] * n
    for i in range(n):
        sum_val = 0.0
        for j in range(i):
            sum_val += L_dense[i][j] * y[j]
        y[i] = b[i] - sum_val
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_val = 0.0
        for j in range(i + 1, n):
            sum_val += U_dense[i][j] * x[j]
        if abs(U_dense[i][i]) < 1e-12:
            return None

        x[i] = (y[i] - sum_val) / U_dense[i][i]

    return x

def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение
    det(A) = det(L) * det(U)
    """
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None
    _, U = lu_result
    n = A.shape[0]
    det = 1.0
    dense_U = U.to_dense()
    for i in range(n):
        det *= dense_U[i][i]

    return det
