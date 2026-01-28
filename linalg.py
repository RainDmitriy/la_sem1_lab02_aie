from CSC import CSCMatrix
from CSR import CSRMatrix
from matrix_types import Vector
from typing import Tuple, Optional


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    Ожидается, что матрица L хранит единицы на главной диагонали.
    """
    n = A.shape[0]
    if n != A.shape[1]: return None

    a = A.to_dense()
    l = [[0.0] * n for _ in range(n)]
    u = [[0.0] * n for _ in range(n)]

    for i in range(n):
        l[i][i] = 1.0
        for j in range(i, n):
            s = sum(l[i][k] * u[k][j] for k in range(i))
            u[i][j] = a[i][j] - s
        for j in range(i + 1, n):
            s = sum(l[j][k] * u[k][i] for k in range(i))
            if abs(u[i][i]) < 1e-12: return None
            l[j][i] = (a[j][i] - s) / u[i][i]

    return CSCMatrix.from_dense(l), CSCMatrix.from_dense(u)


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    res = lu_decomposition(A)
    if not res: return None
    L, U = res
    n = len(b)

    ld = L.to_dense()
    ud = U.to_dense()

    y = [0.0] * n
    for i in range(n):
        s = sum(ld[i][k] * y[k] for k in range(i))
        y[i] = b[i] - s

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(ud[i][k] * x[k] for k in range(i + 1, n))
        if abs(ud[i][i]) < 1e-12: return None
        x[i] = (y[i] - s) / ud[i][i]

    return x


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    res = lu_decomposition(A)
    if not res: return 0.0
    _, U = res

    ud = U.to_dense()
    det = 1.0
    for i in range(len(ud)):
        det *= ud[i][i]

    return det