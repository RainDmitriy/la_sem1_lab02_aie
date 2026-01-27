from CSC import CSCMatrix
from type import Vector
from typing import Tuple, Optional


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    Ожидается, что матрица L хранит единицы на главной диагонали.
    """
    n = A.shape[0]
    if n != A.shape[1]:
        raise ValueError("LU-разложение применимо только к квадратным матрицам")

    dense = A.to_dense()

    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for k in range(i, n):
            sum_ = 0.0
            for j in range(i):
                sum_ += L[i][j] * U[j][k]
            U[i][k] = dense[i][k] - sum_

        L[i][i] = 1.0
        for k in range(i + 1, n):
            sum_ = 0.0
            for j in range(i):
                sum_ += L[k][j] * U[j][i]

            if abs(U[i][i]) < 1e-10:
                return None

            L[k][i] = (dense[k][i] - sum_) / U[i][i]

    from CSC import CSCMatrix
    L_csc = CSCMatrix.from_dense(L)
    U_csc = CSCMatrix.from_dense(U)

    return (L_csc, U_csc)


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None

    L, U = lu_result
    n = len(b)

    y = [0.0] * n
    dense_L = L.to_dense()
    for i in range(n):
        sum_ = 0.0
        for j in range(i):
            sum_ += dense_L[i][j] * y[j]
        y[i] = b[i] - sum_  # L[i][i] = 1

    x = [0.0] * n
    dense_U = U.to_dense()

    for i in range(n - 1, -1, -1):
        sum_ = 0.0
        for j in range(i + 1, n):
            sum_ += dense_U[i][j] * x[j]

        if abs(dense_U[i][i]) < 1e-10:
            return None

        x[i] = (y[i] - sum_) / dense_U[i][i]

    return x


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U) = 1 * произведение диагональных элементов U
    """
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None

    _, U = lu_result
    dense_U = U.to_dense()

    det = 1.0
    n = A.shape[0]

    for i in range(n):
        det *= dense_U[i][i]

    return det