from CSC import CSCMatrix
from types import Vector
from typing import Tuple, Optional


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    Ожидается, что матрица L хранит единицы на главной диагонали.
    """
    # преобразуем в плотный формат
    dense_A = A.to_dense()
    n = A.shape[0]

    if n != A.shape[1]:
        raise ValueError("Матрица должна быть квадратной")

    L_dense = [[0.0] * n for _ in range(n)]
    U_dense = [[0.0] * n for _ in range(n)]

    try:
        for i in range(n):
            # верхняя треугольная матрица U
            for j in range(i, n):
                total = dense_A[i][j]
                for k in range(i):
                    total -= L_dense[i][k] * U_dense[k][j]
                U_dense[i][j] = total

            # нижняя треугольная матрица L
            L_dense[i][i] = 1.0
            for j in range(i + 1, n):
                total = dense_A[j][i]
                for k in range(i):
                    total -= L_dense[j][k] * U_dense[k][i]
                if U_dense[i][i] == 0:
                    return None  # Матрица вырожденная
                L_dense[j][i] = total / U_dense[i][i]
    except ZeroDivisionError:
        return None

    L = CSCMatrix.from_dense(L_dense)
    U = CSCMatrix.from_dense(U_dense)

    return L, U


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    result = lu_decomposition(A)
    if result is None:
        return None

    L, U = result
    n = A.shape[0]

    if len(b) != n:
        raise ValueError("Размер вектора b не совпадает с размером матрицы A")

    dense_L = L.to_dense()
    dense_U = U.to_dense()

    # подстановка: Ly = b
    y = [0.0] * n
    for i in range(n):
        total = b[i]
        for j in range(i):
            total -= dense_L[i][j] * y[j]
        y[i] = total

    # Ux = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        total = y[i]
        for j in range(i + 1, n):
            total -= dense_U[i][j] * x[j]
        if dense_U[i][i] == 0:
            return None
        x[i] = total / dense_U[i][i]

    return x


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    result = lu_decomposition(A)
    if result is None:
        return None

    L, U = result
    n = A.shape[0]
    # преобразуем U в плотный формат
    dense_U = U.to_dense()
    det = 1.0
    for i in range(n):
        det *= dense_U[i][i]

    return det