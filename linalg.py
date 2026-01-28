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
    L = [[0.0 for _ in range(n)] for _ in range(n)]
    U = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            s = 0.0
            for k in range(i):
                s += L[i][k] * U[k][j]
            U[i][j] = dense[i][j] - s
        if U[i][i] == 0.0:
            return None
        for j in range(i, n):
            if i == j:
                L[j][i] = 1.0
            else:
                s = 0.0
                for k in range(i):
                    s += L[j][k] * U[k][i]
                L[j][i] = (dense[j][i] - s) / U[i][i]

    return CSCMatrix.from_dense(L), CSCMatrix.from_dense(U)

def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    decomposition = lu_decomposition(A)
    if decomposition is None:
        return None

    L, U = decomposition
    n = len(b)
    y = [0.0] * n
    L_dense = L.to_dense()
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L_dense[i][j] * y[j]
        y[i] = b[i] - s
    x = [0.0] * n
    U_dense = U.to_dense()
    for i in range(n - 1, -1, -1):
        if U_dense[i][i] == 0.0:
            return None
        s = 0.0
        for j in range(i + 1, n):
            s += U_dense[i][j] * x[j]
        x[i] = (y[i] - s) / U_dense[i][i]

    return x

def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    decomposition = lu_decomposition(A)
    if decomposition is None:
        return None

    _, U = decomposition
    n, _ = U.shape
    U_dense = U.to_dense()
    det = 1.0
    for i in range(n):
        det *= U_dense[i][i]

    return det

