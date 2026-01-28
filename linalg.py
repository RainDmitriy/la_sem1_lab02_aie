# linalg.py
from CSC import CSCMatrix
from types import Vector
from typing import Tuple, Optional

def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    Матрица L хранит единицы на диагонали.
    """
    n = A.shape[0]
    assert A.shape[0] == A.shape[1], "Матрица должна быть квадратной"

    # Переводим в плотный формат для простоты вычислений
    dense = A.to_dense()
    L_dense = [[0.0]*n for _ in range(n)]
    U_dense = [[0.0]*n for _ in range(n)]

    for i in range(n):
        L_dense[i][i] = 1.0

    # Прямой алгоритм без частичной выборки
    for j in range(n):
        for i in range(j+1):
            s = sum(U_dense[k][j] * L_dense[i][k] for k in range(i))
            U_dense[i][j] = dense[i][j] - s

        for i in range(j+1, n):
            s = sum(U_dense[k][j] * L_dense[i][k] for k in range(j))
            if U_dense[j][j] == 0.0:
                U_dense[j][j] = 1e-12  # избегаем деления на ноль
            L_dense[i][j] = (dense[i][j] - s) / U_dense[j][j]

    # Переводим обратно в CSC
    L = CSCMatrix.from_dense(L_dense)
    U = CSCMatrix.from_dense(U_dense)
    return L, U

def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    LU = lu_decomposition(A)
    if LU is None:
        return None
    L, U = LU
    n = len(b)

    # Прямой ход: Ly = b
    y = [0.0]*n
    L_dense = L.to_dense()
    for i in range(n):
        y[i] = b[i] - sum(L_dense[i][j]*y[j] for j in range(i))

    # Обратный ход: Ux = y
    x = [0.0]*n
    U_dense = U.to_dense()
    for i in reversed(range(n)):
        x[i] = y[i] - sum(U_dense[i][j]*x[j] for j in range(i+1, n))
        if U_dense[i][i] == 0.0:
            U_dense[i][i] = 1e-12  # избегаем деления на ноль
        x[i] /= U_dense[i][i]

    return x

def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    LU = lu_decomposition(A)
    if LU is None:
        return None
    _, U = LU
    U_dense = U.to_dense()
    det = 1.0
    for i in range(len(U_dense)):
        det *= U_dense[i][i]
    return det
