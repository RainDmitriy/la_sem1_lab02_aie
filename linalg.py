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
    n, m = A.shape
    if n != m:
        return None

    a = A.to_dense()
    L = [[0.0 for _ in range(n)] for _ in range(n)]
    U = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        L[i][i] = 1.0

    for k in range(n):
        s = 0.0
        for p in range(k):
            s += L[k][p] * U[p][k]
        pivot = a[k][k] - s
        if pivot == 0:
            return None
        U[k][k] = pivot

        for j in range(k + 1, n):
            s = 0.0
            for p in range(k):
                s += L[k][p] * U[p][j]
            U[k][j] = a[k][j] - s

        for i in range(k + 1, n):
            s = 0.0
            for p in range(k):
                s += L[i][p] * U[p][k]
            L[i][k] = (a[i][k] - s) / U[k][k]

    L_csc = CSCMatrix.from_dense(L)
    U_csc = CSCMatrix.from_dense(U)
    return L_csc, U_csc


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    res = lu_decomposition(A)
    if res is None:
        return None
    L, U = res

    n, m = A.shape
    if len(b) != n:
        return None

    Ld = L.to_dense()
    Ud = U.to_dense()

    y = [0.0 for _ in range(n)]
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += Ld[i][j] * y[j]
        y[i] = b[i] - s

    x = [0.0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += Ud[i][j] * x[j]
        if Ud[i][i] == 0:
            return None
        x[i] = (y[i] - s) / Ud[i][i]

    return x
def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    res = lu_decomposition(A)
    if res is None:
        return None
    _, U = res
    Ud = U.to_dense()
    n = len(Ud)
    det = 1.0
    for i in range(n):
        det *= Ud[i][i]
    return det
