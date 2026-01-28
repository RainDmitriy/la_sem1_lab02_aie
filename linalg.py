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

    dense_a = A.to_dense()
    u = [[0.0] * rows for _ in range(rows)]
    l = [[0.0] * rows for _ in range(rows)]

    for i in range(rows):
        l[i][i] = 1.0
    for i in range(rows):
        for m in range(i, rows):
            summa = 0.0
            for j in range(i):
                summa += l[i][j] * u[j][m]
            u[i][m] = dense_a[i][m] - summa
        for m in range(i + 1, rows):
            summa = 0.0
            for j in range(i):
                summa += l[m][j] * u[j][i]

            pivot = u[i][i]

            if abs(pivot) < 1e-8:
                return None

            l[m][i] = (dense_a[m][i]-summa)/pivot

    if abs(u[rows - 1][rows - 1]) < 1e-8:
        return None
    csc_l = CSCMatrix.from_dense(l)
    csc_u = CSCMatrix.from_dense(u)

    return csc_l, csc_u


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

    l, u = lu_decomposition(A)
    n = len(b)
    dense_l = l.to_dense()
    dense_u = u.to_dense()

    y = [0.0] * n
    for i in range(n):
        summa = 0
        for j in range(i):
            summa += l[i][j] * y[j]
        y[i] = b[i] - summa

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        summa = 0
        for j in range(i + 1, n):
            summa += u[i][j] * x[j]
        x[i] = (y[i] - summa) / u[i][i]
    return x


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    det = 1
    n = A.shape[0]
    l, u = lu_decomposition(A)
    dense_u = u.to_dense()
    for i in range(n):
        det *= dense_u[i][i]

    return det
