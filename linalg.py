from CSC import CSCMatrix
from CSR import CSRMatrix
from type import Vector, TOLERANCE
from typing import Tuple, Optional

def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    Ожидается, что матрица L хранит единицы на главной диагонали.
    """
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")

    l_data, l_indices, l_indptr = [], [], [0]
    u_data, u_indices, u_indptr = [], [], [0]
    workspace = [0.0] * n
    active_rows = []

    for j in range(n):
        for idx in range(A.indptr[j], A.indptr[j+1]):
            row = A.indices[idx]
            workspace[row] = A.data[idx]
            active_rows.append(row)

        active_rows.sort()

        for i in range(j):
            if abs(workspace[i]) < TOLERANCE:
                continue

            u_val = workspace[i]
            u_data.append(u_val)
            u_indices.append(i)

            for l_idx in range(l_indptr[i], l_indptr[i+1]):
                l_row = l_indices[l_idx]
                l_val = l_data[l_idx]
                if l_row > i:
                    if workspace[l_row] == 0.0:
                        active_rows.append(l_row)
                    workspace[l_row] -= l_val * u_val

            workspace[i] = 0.0

        diag_val = workspace[j]
        if abs(diag_val) < TOLERANCE:
            return None

        u_data.append(diag_val)
        u_indices.append(j)
        u_indptr.append(len(u_data))

        l_data.append(1.0)
        l_indices.append(j)

        active_rows = sorted(list(set(active_rows)))
        for r in active_rows:
            if r > j and abs(workspace[r]) > TOLERANCE:
                l_data.append(workspace[r] / diag_val)
                l_indices.append(r)
                workspace[r] = 0.0
            else:
                workspace[r] = 0.0

        l_indptr.append(len(l_data))
        active_rows = []

    return (
        CSCMatrix(l_data, l_indices, l_indptr, A.shape),
        CSCMatrix(u_data, u_indices, u_indptr, A.shape)
    )

def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    res = lu_decomposition(A)
    if not res: return None
    L, U = res
    n = len(b)

    y = list(b)
    for j in range(n):
        for idx in range(L.indptr[j], L.indptr[j+1]):
            row = L.indices[idx]
            if row > j:
                y[row] -= L.data[idx] * y[j]

    x = list(y)
    for j in range(n - 1, -1, -1):
        diag_val = 0.0
        for idx in range(U.indptr[j], U.indptr[j+1]):
            if U.indices[idx] == j:
                diag_val = U.data[idx]
                break

        x[j] /= diag_val
        for idx in range(U.indptr[j], U.indptr[j+1]):
            row = U.indices[idx]
            if row < j:
                x[row] -= U.data[idx] * x[j]

    return x

def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    res = lu_decomposition(A)
    if not res: return 0.0
    _, U = res
    det = 1.0
    for j in range(U.shape[1]):
        for idx in range(U.indptr[j], U.indptr[j+1]):
            if U.indices[idx] == j:
                det *= U.data[idx]
    return det