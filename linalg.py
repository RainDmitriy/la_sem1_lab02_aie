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
    l = [[0.0] * rows for _ in range(rows)]
    u = [[0.0] * rows for _ in range(rows)]
    n = rows

    for i in range(n):
        for j in range(i, n):
            u[i][j] = a[i][j] - sum(l[i][k] * u[k][j] for k in range(i))

        if abs(u[i][i]) < 1e-12:
            return None

        for j in range(i + 1, n):
            l[j][i] = (a[j][i] - sum(l[j][k] * u[k][i] for k in range(i))) / u[i][i]

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
    lu = lu_decomposition(A)
    if lu is None:
        return None

    l, u = lu
    n = len(b)
    dense_l = l.to_dense()
    dense_u = u.to_dense()

    y = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += dense_l[i][j] * y[j]
        y[i] = b[i] - s

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(dense_u[i][i]) < 1e-12:
            return None
        s = 0.0
        for j in range(i + 1, n):
            s += dense_u[i][j] * x[j]
        x[i] = (y[i] - s) / dense_u[i][i]

    return x


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    lu = lu_decomposition(A)
    if lu is None:
        return None

    l, u = lu
    dense_u = u.to_dense()
    det = 1.0
    for i in range(len(dense_u)):
        det *= dense_u[i][i]
    return det
