from CSC import CSCMatrix
from CSR import CSRMatrix
from types import Vector
from typing import Tuple, Optional

def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    Ожидается, что матрица L хранит единицы на главной диагонали.
    """
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        return None
    A_dense = A.to_dense()
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = 1.0
        for j in range(i, n):
            sum_val = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = A_dense[i][j] - sum_val
        for j in range(i + 1, n):
            if U[i][i] == 0: return None
            sum_val = sum(L[j][k] * U[k][i] for k in range(i))
            L[j][i] = (A_dense[j][i] - sum_val) / U[i][i]
    return (CSCMatrix.from_dense(L), CSCMatrix.from_dense(U))

def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    res = lu_decomposition(A)
    if not res: return None
    L, U = res[0].to_dense(), res[1].to_dense()
    n = len(b)
    y = [0.0] * n
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if U[i][i] == 0: return None
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x

def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    res = lu_decomposition(A)
    if not res: return 0.0
    U = res[1].to_dense()
    det = 1.0
    for i in range(len(U)):
        det *= U[i][i]
    return det

