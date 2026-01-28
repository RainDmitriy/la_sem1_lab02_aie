from CSC import CSCMatrix
from CSR import CSRMatrix
from type import Vector
from typing import Tuple, Optional

EPSILON = 1e-10


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    Ожидается, что матрица L хранит единицы на главной диагонали.
    """
    n_rows, n_cols = A.shape
    if n_rows != n_cols:
        raise ValueError("LU‑разложение определено только для квадратных матриц")
    n = n_rows
    dense_A = A.to_dense()
    L: list[list[float]] = [[0.0] * n for _ in range(n)]
    U: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            sum_u = 0.0
            for k in range(i):
                sum_u += L[i][k] * U[k][j]
            U[i][j] = dense_A[i][j] - sum_u
        if abs(U[i][i]) < EPSILON:
            return None
        L[i][i] = 1.0
        for j in range(i + 1, n):
            sum_l = 0.0
            for k in range(i):
                sum_l += L[j][k] * U[k][i]
            L[j][i] = (dense_A[j][i] - sum_l) / U[i][i]
    L_csc = CSCMatrix.from_dense(L)
    U_csc = CSCMatrix.from_dense(U)
    return L_csc, U_csc


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    lu = lu_decomposition(A)
    if lu is None:
        return None
    L, U = lu
    n = A.shape[0]
    if len(b) != n:
        raise ValueError("Размерность вектора b должна соответствовать матрице")
    L_dense = L.to_dense()
    U_dense = U.to_dense()
    y: Vector = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L_dense[i][j] * y[j]
        y[i] = b[i] - s
    x: Vector = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += U_dense[i][j] * x[j]
        if abs(U_dense[i][i]) < EPSILON:
            return None
        x[i] = (y[i] - s) / U_dense[i][i]
    return x


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    lu = lu_decomposition(A)
    if lu is None:
        return None
    _, U = lu
    n = A.shape[0]
    U_dense = U.to_dense()
    det = 1.0
    for i in range(n):
        det *= U_dense[i][i]
    return det