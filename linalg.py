from CSC import CSCMatrix
from matrix_types import Vector
from typing import Tuple, Optional

def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы
    Ожидается, что матрица L хранит единицы на главной диагонали
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("LU-разложение определено только для квадратных матриц")
    n = A.shape[0]
    dense = A.to_dense()
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = 1.0
    for i in range(n):
        for k in range(i, n):
            sum_val = 0.0
            for j in range(i):
                sum_val += L[i][j] * U[j][k]
            U[i][k] = dense[i][k] - sum_val
        for k in range(i + 1, n):
            sum_val = 0.0
            for j in range(i):
                sum_val += L[k][j] * U[j][i]
            if abs(U[i][i]) < 1e-12:
                return None
            L[k][i] = (dense[k][i] - sum_val) / U[i][i]
    L_csc = CSCMatrix.from_dense(L)
    U_csc = CSCMatrix.from_dense(U)
    return L_csc, U_csc

def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None
    L, U = lu_result
    n = len(b)
    y = [0.0] * n
    for i in range(n):
        sum_val = 0.0
        for j in range(i):
            sum_val += L.to_dense()[i][j] * y[j]
        y[i] = b[i] - sum_val
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_val = 0.0
        for j in range(i + 1, n):
            sum_val += U.to_dense()[i][j] * x[j]
        if abs(U.to_dense()[i][i]) < 1e-12:
            return None
        x[i] = (y[i] - sum_val) / U.to_dense()[i][i]

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
