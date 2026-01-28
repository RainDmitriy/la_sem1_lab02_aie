from CSC import CSCMatrix
from CSR import CSRMatrix
from type import Vector
from typing import Tuple, Optional

def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    Ожидается, что матрица L хранит единицы на главной диагонали.
    """
    dense_A = A.to_dense()
    n = len(dense_A)
    if n == 0 or len(dense_A[0]) != n:
        return None
    L_dense = [[0.0] * n for _ in range(n)]
    U_dense = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            sum_val = 0.0
            for k in range(i):
                sum_val += L_dense[i][k] * U_dense[k][j]
            U_dense[i][j] = dense_A[i][j] - sum_val
        if abs(U_dense[i][i]) < 1e-12:
            return None
        L_dense[i][i] = 1.0
        for j in range(i + 1, n):
            sum_val = 0.0
            for k in range(i):
                sum_val += L_dense[j][k] * U_dense[k][i]
            L_dense[j][i] = (dense_A[j][i] - sum_val) / U_dense[i][i]
    L = CSCMatrix.from_dense(L_dense)
    U = CSCMatrix.from_dense(U_dense)
    return L, U

def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None
    L, U = lu_result
    n = len(b)
    if n != A.shape[0]:
        return None
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
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None
    L, U = lu_result
    U_dense = U.to_dense()
    n = len(U_dense)
    det = 1.0
    for i in range(n):
        det *= U_dense[i][i]
    return det

