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
    if A.shape[0] != A.shape[1]:
        raise ValueError("LU-разложение возможно только для квадратных матриц")

    a_dense = A.to_dense()
    l_dense = [[0.0] * n for _ in range(n)]
    u_dense = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for k in range(i, n):
            sum_u = 0.0
            for p in range(i):
                sum_u += l_dense[i][p] * u_dense[p][k]
            u_dense[i][k] = a_dense[i][k] - sum_u

        if abs(u_dense[i][i]) < 1e-12:
            return None

        for k in range(i, n):
            if i == k:
                l_dense[i][i] = 1.0
            else:
                sum_l = 0.0
                for p in range(i):
                    sum_l += l_dense[k][p] * u_dense[p][i]
                l_dense[k][i] = (a_dense[k][i] - sum_l) / u_dense[i][i]

    L = CSCMatrix.from_dense(l_dense)
    U = CSCMatrix.from_dense(u_dense)

    return L, U

def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    lu_res = lu_decomposition(A)
    if lu_res is None:
        return None

    L, U = lu_res
    n = len(b)
    y = list(b)

    for j in range(n):
        current_y = y[j]
        start_ptr = L.indptr[j]
        end_ptr = L.indptr[j+1]

        for idx in range(start_ptr, end_ptr):
            row_idx = L.indices[idx]
            val = L.data[idx]
            if row_idx > j:
                y[row_idx] -= val * current_y

    x = list(y)
    for j in range(n - 1, -1, -1):
        u_ii = 0.0
        start_ptr = U.indptr[j]
        end_ptr = U.indptr[j+1]

        for idx in range(start_ptr, end_ptr):
            if U.indices[idx] == j:
                u_ii = U.data[idx]
                break

        if abs(u_ii) < 1e-12:
            return None

        x[j] /= u_ii
        current_x = x[j]

        for idx in range(start_ptr, end_ptr):
            row_idx = U.indices[idx]
            val = U.data[idx]
            if row_idx < j:
                x[row_idx] -= val * current_x

    return x

def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    lu_res = lu_decomposition(A)
    if lu_res is None:
        return 0.0

    _, U = lu_res
    det = 1.0
    n = U.shape[0]

    for j in range(n):
        diag_val = 0.0
        for idx in range(U.indptr[j], U.indptr[j+1]):
            if U.indices[idx] == j:
                diag_val = U.data[idx]
                break
        det *= diag_val

    return det