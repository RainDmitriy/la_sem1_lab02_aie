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
        max_idx = k
        max_val = abs(a[k][k])
        for i in range(k + 1, n):
            if abs(a[i][k]) > max_val:
                max_val = abs(a[i][k])
                max_idx = i
        
        if max_val < 1e-10:
            return None
        
        if max_idx != k:
            a[k], a[max_idx] = a[max_idx], a[k]
            for j in range(k):
                L[k][j], L[max_idx][j] = L[max_idx][j], L[k][j]
        
        U_k = U[k]
        L_k = L[k]
        a_k = a[k]
        for j in range(k, n):
            s = 0.0
            for p in range(k):
                s += L_k[p] * U[p][j]
            U_k[j] = a_k[j] - s

        pivot = U_k[k]
        if abs(pivot) < 1e-10:
            return None

        for i in range(k + 1, n):
            s = 0.0
            for p in range(k):
                s += L[i][p] * U[p][k]
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
        s = 0.0
        L_i = L[i]
        for j in range(i):
            s += L_i[j] * y[j]
        if abs(L_i[i]) < 1e-10:
            return None
        y[i] = (b[i] - s) / L_i[i]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = 0.0
        U_i = U[i]
        for j in range(i + 1, n):
            s += U_i[j] * x[j]
        if abs(U_i[i]) < 1e-10:
            return None
        x[i] = (y[i] - s) / U_i[i]

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
