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
    n, m = A.shape
    if n != m:
        return None
    dense = A.to_dense()
    L = [[0.0]*n for _ in range(n)]
    U = [[0.0]*n for _ in range(n)]
    for i in range(n):
        L[i][i] = 1.0
    for k in range(n):
        for j in range(k, n):
            s = sum(L[k][p] * U[p][j] for p in range(k))
            U[k][j] = dense[k][j] - s
        if U[k][k] == 0:
            return None
        for i in range(k+1, n):
            s = sum(L[i][p] * U[p][k] for p in range(k))
            L[i][k] = (dense[i][k] - s) / U[k][k]
    return CSCMatrix.from_dense(L), CSCMatrix.from_dense(U)

def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    result = lu_decomposition(A)
    if result is None:
        return None
    L, U = result
    Ld = L.to_dense()
    Ud = U.to_dense()
    n = len(Ld)
    b = list(b)

    y = [0.0] * n
    for i in range(n):
        s = b[i]
        for j in range(i):
            s -= Ld[i][j] * y[j]
        y[i] = s
    x = [0.0] * n
    for i in reversed(range(n)):
        s = y[i]
        for j in range(i + 1, n):
            s -= Ud[i][j] * x[j]
        if Ud[i][i] == 0:
            return None
        x[i] = s / Ud[i][i]

    return x

def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    result = lu_decomposition(A)
    if result is None:
        return None
    _, U = result
    Ud = U.to_dense()
    det = 1.0
    for i in range(len(Ud)):
        det *= Ud[i][i]
    return det
