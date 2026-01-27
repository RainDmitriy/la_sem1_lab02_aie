from CSC import CSCMatrix
from type import Vector
from typing import Tuple, Optional


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    '''
    LU-разложение для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    Ожидается, что матрица L хранит единицы на главной диагонали.
    '''
    n = A.shape[0]
    if n != A.shape[1]:
        return None

    dense = A.to_dense()

    for k in range(n):
        if abs(dense[k][k]) < 1e-12:
            return None

        for i in range(k + 1, n):
            dense[i][k] /= dense[k][k]
            for j in range(k + 1, n):
                dense[i][j] -= dense[i][k] * dense[k][j]

    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i > j:
                L[i][j] = dense[i][j]
                U[i][j] = 0.0
            elif i == j:
                L[i][j] = 1.0
                U[i][j] = dense[i][j]
            else:
                L[i][j] = 0.0
                U[i][j] = dense[i][j]

    from COO import COOMatrix
    L_csc = COOMatrix.from_dense(L)._to_csc()
    U_csc = COOMatrix.from_dense(U)._to_csc()

    return (L_csc, U_csc)


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    '''
    Решение СЛАУ Ax = b через LU-разложение.
    '''
    result = lu_decomposition(A)
    if result is None:
        return None

    L, U = result
    dense_L = L.to_dense()
    dense_U = U.to_dense()
    n = len(b)

    y = [0.0] * n
    for i in range(n):
        sum_val = 0.0
        for j in range(i):
            sum_val += dense_L[i][j] * y[j]
        y[i] = b[i] - sum_val

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_val = 0.0
        for j in range(i + 1, n):
            sum_val += dense_U[i][j] * x[j]

        if abs(dense_U[i][i]) < 1e-12:
            return None

        x[i] = (y[i] - sum_val) / dense_U[i][i]

    return x


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    '''
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    '''
    result = lu_decomposition(A)
    if result is None:
        return None

    _, U = result
    dense_U = U.to_dense()
    det = 1.0
    n = A.shape[0]

    for i in range(n):
        det *= dense_U[i][i]

    return det