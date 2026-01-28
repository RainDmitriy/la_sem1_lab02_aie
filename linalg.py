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
    pivots = list(range(rows))
    l = [[0.0] * rows for _ in range(rows)]
    u = [[0.0] * rows for _ in range(rows)]

    for i in range(rows):
        p_row = max(range(i, rows), key=lambda r: abs(a[r][i]))
        if abs(a[p_row][i]) < 1e-12:
            return None

        a[i], a[p_row] = a[p_row], a[i]
        pivots[i], pivots[p_row] = pivots[p_row], pivots[i]
        l[i][i] = 1.0

        for j in range(i, rows):
            u[i][j] = a[i][j] - sum(l[i][k] * u[k][j] for k in range(i))
        for j in range(i + 1, rows):
            l[j][i] = (a[j][i] - sum(l[j][k] * u[k][i] for k in range(i))) / u[i][i]

    return CSCMatrix.from_dense(l), CSCMatrix.from_dense(u), pivots


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

    res = lu_decomposition(A)
    if res is None:
        return None

    l, u, pivots = res
    n = len(b)
    b_perm = [b[pivots[i]] for i in range(n)]
    dense_l = l.to_dense()
    dense_u = u.to_dense()

    y = [0.0] * n
    for i in range(n):
        y[i] = b_perm[i] - sum(dense_l[i][j] * y[j] for j in range(i))

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(dense_u[i][i]) < 1e-12:
            return None
        x[i] = (y[i] - sum(dense_u[i][j] * x[j] for j in range(i + 1, n))) / dense_u[i][i]

    return x


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    lu = lu_decomposition(A)
    if lu is None:
        return None

    l, u, pivots = lu
    dense_u = u.to_dense()
    det = 1.0

    for i in range(len(dense_u)):
        det *= dense_u[i][i]

    swaps = sum(1 for i in range(len(pivots)) if pivots[i] != i)
    if swaps % 2:
        det *= -1

    return det
