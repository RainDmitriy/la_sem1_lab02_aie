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

    if n == 0 or any(len(row) != n for row in dense):
        return None

    a = [row[:] for row in dense]

    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        L[i][i] = 1.0

    for k in range(n):
        for j in range(k, n):
            s = sum(L[k][p] * U[p][j] for p in range(k))
            U[k][j] = a[k][j] - s

        pivot = U[k][k]
        if pivot == 0.0:
            return None

        for i in range(k + 1, n):
            s = sum(L[i][p] * U[p][k] for p in range(k))
            L[i][k] = (a[i][k] - s) / pivot

    return CSCMatrix.from_dense(L), CSCMatrix.from_dense(U)

def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    lu = lu_decomposition(A)
    if lu is None:
        return None

    L, U = lu
    L = L.to_dense()
    U = U.to_dense()

    n = len(L)
    if len(b) != n:
        return None

    y = [0.0] * n
    for i in range(n):
        s = sum(L[i][j] * y[j] for j in range(i))
        if L[i][i] == 0.0:
            return None
        y[i] = (b[i] - s) / L[i][i]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(U[i][j] * x[j] for j in range(i + 1, n))
        if U[i][i] == 0.0:
            return None
        x[i] = (y[i] - s) / U[i][i]

    return x

def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    lu = lu_decomposition(A)
    if lu is None:
        return None

    U = lu[1].to_dense()
    det = 1.0
    for i in range(len(U)):
        det *= U[i][i]

    return det

