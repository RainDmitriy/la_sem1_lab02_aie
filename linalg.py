from type import Vector
from typing import Tuple, Optional, List
from CSC import CSCMatrix


EPSILON = 1e-12


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """LU-разложение через алгоритм Краута с разреженными структурами."""
    n = A.shape[0]

    if n != A.shape[1]:
        raise ValueError("Матрица должна быть квадратной")

    A_rows = [{} for _ in range(n)]
    for j in range(n):
        start, end = A.indptr[j], A.indptr[j + 1]
        for pos in range(start, end):
            i = A.indices[pos]
            A_rows[i][j] = A.data[pos]

    L_cols = [{} for _ in range(n)]
    U_rows = [{} for _ in range(n)]

    active_updates = [{} for _ in range(n)]

    for k in range(n):
        u_row = A_rows[k].copy()

        for j, val in active_updates[k].items():
            u_row[j] = u_row.get(j, 0.0) + val

        u_row = {j: val for j, val in u_row.items() if j >= k}

        u_kk = u_row.get(k, 0.0)

        if abs(u_kk) < EPSILON:
            return None

        U_rows[k] = {j: val for j, val in u_row.items() if abs(val) > EPSILON}

        L_cols[k][k] = 1.0

        for i in range(k + 1, n):
            a_ik = A_rows[i].get(k, 0.0)

            update_ik = active_updates[i].get(k, 0.0)
            total = a_ik + update_ik

            if abs(total) > EPSILON:
                l_ik = total / u_kk
                L_cols[k][i] = l_ik

                for j, u_kj in u_row.items():
                    if j > k:
                        update = -l_ik * u_kj
                        if abs(update) > EPSILON:
                            active_updates[i][j] = active_updates[i].get(j, 0.0) + update

    L_data, L_indices, L_indptr = [], [], [0]
    for j in range(n):
        rows = sorted(L_cols[j].keys())
        for i in rows:
            L_data.append(L_cols[j][i])
            L_indices.append(i)
        L_indptr.append(len(L_data))

    U_cols = [{} for _ in range(n)]
    for i in range(n):
        for j, val in U_rows[i].items():
            U_cols[j][i] = val

    U_data, U_indices, U_indptr = [], [], [0]
    for j in range(n):
        rows = sorted(U_cols[j].keys())
        for i in rows:
            U_data.append(U_cols[j][i])
            U_indices.append(i)
        U_indptr.append(len(U_data))

    L = CSCMatrix(L_data, L_indices, L_indptr, (n, n))
    U = CSCMatrix(U_data, U_indices, U_indptr, (n, n))

    return L, U


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """Решение СЛАУ Ax = b через LU-разложение."""
    result = lu_decomposition(A)
    if result is None:
        return None

    L, U = result
    n = A.shape[0]

    if len(b) != n:
        raise ValueError("Размер вектора b не совпадает с размером матрицы A")

    y = [0.0] * n
    L_rows = [{} for _ in range(n)]
    for j in range(n):
        start, end = L.indptr[j], L.indptr[j + 1]
        for pos in range(start, end):
            i = L.indices[pos]
            L_rows[i][j] = L.data[pos]

    for i in range(n):
        total = b[i]
        for j, val in L_rows[i].items():
            if j < i:
                total -= val * y[j]
        y[i] = total

    U_rows = [{} for _ in range(n)]
    for j in range(n):
        start, end = U.indptr[j], U.indptr[j + 1]
        for pos in range(start, end):
            i = U.indices[pos]
            U_rows[i][j] = U.data[pos]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        total = y[i]
        for j, val in U_rows[i].items():
            if j > i:
                total -= val * x[j]
        u_ii = U_rows[i].get(i, 0.0)
        if abs(u_ii) < EPSILON:
            return None
        x[i] = total / u_ii

    return x


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """Нахождение определителя через LU-разложение."""
    result = lu_decomposition(A)
    if result is None:
        return None

    L, U = result
    n = A.shape[0]
    det = 1.0
    for j in range(n):
        start, end = U.indptr[j], U.indptr[j + 1]
        for pos in range(start, end):
            if U.indices[pos] == j:
                det *= U.data[pos]
                break
        else:
            det = 0.0
            break

    return det