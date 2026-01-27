from CSC import CSCMatrix
from type import Vector
from typing import Tuple, Optional, List


def lu_decomposition_pivot(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix, List[int]]]:
    """LU-разложение с частичным выбором ведущего элемента."""
    n = A.shape[0]
    if n != A.shape[1]:
        raise ValueError("LU-разложение применимо только к квадратным матрицам")

    dense = A.to_dense()

    perm = list(range(n))

    LU = [row[:] for row in dense]

    for i in range(n):
        max_val = abs(LU[i][i])
        max_row = i
        for k in range(i + 1, n):
            if abs(LU[k][i]) > max_val:
                max_val = abs(LU[k][i])
                max_row = k

        if max_val < 1e-12:
            return None

        if max_row != i:
            LU[i], LU[max_row] = LU[max_row], LU[i]
            perm[i], perm[max_row] = perm[max_row], perm[i]

        for k in range(i + 1, n):
            if abs(LU[i][i]) < 1e-12:
                return None
            LU[k][i] /= LU[i][i]
            for j in range(i + 1, n):
                LU[k][j] -= LU[k][i] * LU[i][j]

    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i > j:
                L[i][j] = LU[i][j]
                U[i][j] = 0.0
            elif i == j:
                L[i][j] = 1.0
                U[i][j] = LU[i][j]
            else:
                L[i][j] = 0.0
                U[i][j] = LU[i][j]

    L_csc = CSCMatrix.from_dense(L)
    U_csc = CSCMatrix.from_dense(U)

    return (L_csc, U_csc, perm)


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """LU-разложение без возврата перестановок (для совместимости)."""
    result = lu_decomposition_pivot(A)
    if result is None:
        return None
    L, U, _ = result
    return (L, U)


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """Решение СЛАУ Ax = b методом LU-разложения с выбором ведущего элемента."""
    result = lu_decomposition_pivot(A)
    if result is None:
        return None

    L, U, perm = result
    n = len(b)

    b_perm = [b[perm[i]] for i in range(n)]

    dense_L = L.to_dense()
    dense_U = U.to_dense()

    y = [0.0] * n
    for i in range(n):
        sum_val = 0.0
        for j in range(i):
            sum_val += dense_L[i][j] * y[j]
        y[i] = b_perm[i] - sum_val

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
    """Вычисление определителя через LU-разложение."""
    result = lu_decomposition_pivot(A)
    if result is None:
        return None

    _, U, perm = result
    dense_U = U.to_dense()

    det = 1.0
    n = A.shape[0]

    for i in range(n):
        det *= dense_U[i][i]

    sign = 1
    perm_copy = perm[:]
    for i in range(n):
        if perm_copy[i] != i:
            for j in range(i + 1, n):
                if perm_copy[j] == i:
                    perm_copy[i], perm_copy[j] = perm_copy[j], perm_copy[i]
                    sign = -sign
                    break

    return det * sign