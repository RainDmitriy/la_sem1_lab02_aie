from CSC import CSCMatrix
from CSR import CSRMatrix
from types import Vector
from typing import Tuple, Optional


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    L хранит единицы на главной диагонали.
    """
    dense_A = A.to_dense()
    n = len(dense_A)
    if len(dense_A) != len(dense_A[0]):
        return None  # Не квадратная
    L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    U = [row[:] for row in dense_A]
    for k in range(n):
        if abs(U[k][k]) < 1e-12:
            return None  # Нулевой pivot
        for i in range(k + 1, n):
            if abs(U[i][k]) > 1e-12:
                factor = U[i][k] / U[k][k]
                L[i][k] = factor
                for j in range(k, n):
                    U[i][j] -= factor * U[k][j]
    L_csc = CSCMatrix.from_dense(L)
    U_csc = CSCMatrix.from_dense(U)
    return L_csc, U_csc


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    lu = lu_decomposition(A)
    if lu is None:
        return None
    L, U = lu
    dense_b = b[:]
    n = len(dense_b)
    # Forward substitution Ly = b
    y = [0.0] * n
    for i in range(n):
        s = sum(L[i][j] * y[j] for j in range(i))
        y[i] = dense_b[i] - s
    # Back substitution Ux = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - s) / U[i][i]
    return x


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U) = prod(diag(U)), т.к. det(L)=1
    """
    lu = lu_decomposition(A)
    if lu is None:
        return None
    _, U = lu
    det = 1.0
    for i in range(len(U.to_dense())):
        d = U.to_dense()[i][i]
        if abs(d) < 1e-12:
            return 0.0
        det *= d
    return det
