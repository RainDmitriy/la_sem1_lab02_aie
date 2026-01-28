from CSC import CSCMatrix
from CSR import CSRMatrix
from types import Vector
from typing import Tuple, Optional


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    Ожидается, что матрица L хранит единицы на главной диагонали.
    """
    n = A.shape[0]
    assert A.shape[0] == A.shape[1], "Matrix must be square for LU decomposition"

    dense_A = A.to_dense()

    L = [[0.0 for _ in range(n)] for _ in range(n)]
    U = [[0.0 for _ in range(n)] for _ in range(n)]

    P = list(range(n))

    for i in range(n):
        max_row = i
        max_val = abs(dense_A[i][i])

        for k in range(i + 1, n):
            if abs(dense_A[k][i]) > max_val:
                max_val = abs(dense_A[k][i])
                max_row = k

        if max_val < 1e-10:
            return None

        if max_row != i:
            P[i], P[max_row] = P[max_row], P[i]
            dense_A[i], dense_A[max_row] = dense_A[max_row], dense_A[i]
            L[i], L[max_row] = L[max_row], L[i]

        for j in range(i, n):
            sum_u = 0.0
            for k in range(i):
                sum_u += L[i][k] * U[k][j]
            U[i][j] = dense_A[i][j] - sum_u

        if abs(U[i][i]) < 1e-10:
            return None

        for j in range(i + 1, n):
            sum_l = 0.0
            for k in range(i):
                sum_l += L[j][k] * U[k][i]
            L[j][i] = (dense_A[j][i] - sum_l) / U[i][i]

        L[i][i] = 1.0

    L_data, L_rows, L_cols = [], [], []
    U_data, U_rows, U_cols = [], [], []

    for i in range(n):
        for j in range(n):
            if j <= i and abs(L[i][j]) >= 1e-10:
                L_data.append(L[i][j])
                L_rows.append(i)
                L_cols.append(j)

            if j >= i and abs(U[i][j]) >= 1e-10:
                U_data.append(U[i][j])
                U_rows.append(i)
                U_cols.append(j)

    from COO import COOMatrix

    L_coo = COOMatrix(L_data, L_rows, L_cols, (n, n))
    U_coo = COOMatrix(U_data, U_rows, U_cols, (n, n))

    return L_coo._to_csc(), U_coo._to_csc()


def solve_lower_triangular(L: CSCMatrix, b: Vector) -> Vector:
    """Решение Ly = b (L - нижняя треугольная с единицами на диагонали)."""
    n = L.shape[0]
    y = [0.0] * n

    for i in range(n):
        sum_val = 0.0
        row_start = L.indptr[i]
        row_end = L.indptr[i + 1]

        for idx in range(row_start, row_end):
            j = L.indices[idx]
            if j < i:
                sum_val += L.data[idx] * y[j]

        y[i] = b[i] - sum_val

    return y


def solve_upper_triangular(U: CSCMatrix, y: Vector) -> Vector:
    """Решение Ux = y (U - верхняя треугольная)."""
    n = U.shape[0]
    x = [0.0] * n

    for i in range(n - 1, -1, -1):
        sum_val = 0.0
        row_start = U.indptr[i]
        row_end = U.indptr[i + 1]

        for idx in range(row_start, row_end):
            j = U.indices[idx]
            if j > i:
                sum_val += U.data[idx] * x[j]

        diag_val = 0.0
        for idx in range(row_start, row_end):
            if U.indices[idx] == i:
                diag_val = U.data[idx]
                break

        if abs(diag_val) < 1e-10:
            raise ValueError("Matrix U is singular")

        x[i] = (y[i] - sum_val) / diag_val

    return x


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None

    L, U = lu_result

    y = solve_lower_triangular(L, b)

    x = solve_upper_triangular(U, y)

    return x


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None

    L, U = lu_result
    n = A.shape[0]

    det = 1.0

    for i in range(n):
        row_start = U.indptr[i]
        row_end = U.indptr[i + 1]
        diag_val = 0.0

        for idx in range(row_start, row_end):
            if U.indices[idx] == i:
                diag_val = U.data[idx]
                break

        det *= diag_val

    return det
