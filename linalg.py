from CSC import CSCMatrix
# from CSR import CSRMatrix
from type import Vector
from typing import Tuple, Optional


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    Ожидается, что матрица L хранит единицы на главной диагонали.
    """
    rows, cols = A.shape
    if rows != cols:
        raise ValueError("Матрица должна быть квадратной")

    a = A.to_dense()
    n = rows
    l = [[0.0] * n for _ in range(n)]
    u = [[0.0] * n for _ in range(n)]
    for i in range(n):
        a_i = a[i]
        u_i = u[i]
        for j in range(i, n):
            s = 0.0
            for k in range(i):
                s += l[i][k] * u[k][j]
            u_i[j] = a_i[j] - s

        if abs(u[i][i]) < 1e-12:
            return None

        for j in range(i + 1, n):
            s = 0.0
            a_j = a[j]
            for k in range(i):
                s += l[j][k] * u[k][i]
            l[j][i] = (a_j[i] - s) / u[i][i]
        l[i][i] = 1.0

    return CSCMatrix.from_dense(l), CSCMatrix.from_dense(u)


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    LUx = b
    Берем Ux = y
    Решаем Ly = b
    Решаем Ux = y
    """
    if A.shape[0] != len(b):
        raise ValueError("Размерность A должна совпадать с размером b")

    rows, cols = A.shape
    if rows != cols:
        raise ValueError("Матрица должна быть квадратной")
    a = A.to_dense()
    n = rows
    l = [[0.0] * n for _ in range(n)]
    u = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i, n):
            s = 0.0
            for k in range(i):
                s += l[i][k] * u[k][j]
            u[i][j] = a[i][j] - s
        if abs(u[i][i]) < 1e-12:
            return None
        for j in range(i + 1, n):
            s = 0.0
            for k in range(i):
                s += l[j][k] * u[k][i]
            l[j][i] = (a[j][i] - s) / u[i][i]
        l[i][i] = 1.0

    y = [0.0] * n
    for i in range(n):
        s = 0.0
        l_i = l[i]
        for j in range(i):
            s += l_i[j] * y[j]
        y[i] = b[i] - s

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        diag = u[i][i]
        if abs(diag) < 1e-12:
            return None
        s = 0.0
        u_i = u[i]
        for j in range(i + 1, n):
            s += u_i[j] * x[j]
        x[i] = (y[i] - s) / diag

    return x


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    rows, cols = A.shape
    if rows != cols:
        raise ValueError("Матрица должна быть квадратной")

    lu = lu_decomposition(A)
    if lu is None:
        return None

    l, u = lu
    dense_u = u.to_dense()
    det = 1.0
    n = len(dense_u)
    for i in range(n):
        det *= dense_u[i][i]
    return det
