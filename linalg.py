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
    dense = A.to_dense()
    n = len(dense)
    L = [[0.0 for _ in range(n)] for _ in range(n)]
    U = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        L[i][i] = 1.0
    for k in range(n):
        for j in range(k, n):
            s = sum(L[k][t] * U[t][j] for t in range(k))
            U[k][j] = dense[k][j] - s
        if U[k][k] == 0:
            return None
        for i in range(k + 1, n):
            s = sum(L[i][t] * U[t][k] for t in range(k))
            L[i][k] = (dense[i][k] - s) / U[k][k]
    return CSCMatrix.from_dense(L), CSCMatrix.from_dense(U)


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    lu = lu_decomposition(A)
    if lu is None:
        return None
    L, U = lu
    Ld = L.to_dense()
    Ud = U.to_dense()
    n = len(b)
    y = [0.0] * n
    for i in range(n):
        y[i] = b[i] - sum(Ld[i][j] * y[j] for j in range(i))
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if Ud[i][i] == 0:
            return None
        x[i] = (y[i] - sum(Ud[i][j] * x[j] for j in range(i + 1, n))) / Ud[i][i]
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
    Ud = U.to_dense()
    det = 1.0
    for i in range(len(Ud)):
        det *= Ud[i][i]
    return det
