from CSC import CSCMatrix
from CSR import CSRMatrix
from type import Vector
from typing import Tuple, Optional

def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    Ожидается, что матрица L хранит единицы на главной диагонали.
    """
    n_rows, n_cols = A.shape
    if n_rows != n_cols:
        return None
    n = n_rows
    L_data = []
    L_indices = []
    L_indptr = [0] * (n + 1)
    U_data = []
    U_indices = []
    U_indptr = [0] * (n + 1)
    L_temp = [[] for _ in range(n)]
    U_temp = [[] for _ in range(n)]
    A_columns = []
    for j in range(n):
        col_data = []
        for k in range(A.indptr[j], A.indptr[j + 1]):
            i = A.indices[k]
            col_data.append((i, A.data[k]))
        A_columns.append(col_data)
    for k in range(n):
        u_kk = 0.0
        found = False
        for i, val in A_columns[k]:
            if i == k:
                u_kk = val
                found = True
                break
        if not found or abs(u_kk) < 1e-12:
            return None
        U_temp[k].append((k, u_kk))
        for i, val in A_columns[k]:
            if i > k:
                l_ik = val / u_kk
                L_temp[i].append((k, l_ik))
                for j in range(k + 1, n):
                    u_kj = 0.0
                    for idx, u_val in U_temp[k]:
                        if idx == j:
                            u_kj = u_val
                            break
                    if abs(u_kj) > 1e-12:
                        updated = False
                        for idx, (row_idx, _) in enumerate(A_columns[j]):
                            if row_idx == i:
                                A_columns[j][idx] = (i, A_columns[j][idx][1] - l_ik * u_kj)
                                updated = True
                                break
                        if not updated:
                            A_columns[j].append((i, -l_ik * u_kj))
        for j in range(k + 1, n):
            a_kj = 0.0
            for i, val in A_columns[j]:
                if i == k:
                    a_kj = val
                    break
            if abs(a_kj) > 1e-12:
                U_temp[k].append((j, a_kj))
    current_ptr = 0
    for j in range(n):
        L_indptr[j] = current_ptr
        for i in range(j + 1, n):
            for col_idx, val in L_temp[i]:
                if col_idx == j:
                    L_data.append(val)
                    L_indices.append(i)
                    current_ptr += 1
                    break
    L_indptr[n] = current_ptr
    L_with_diag_data = L_data.copy()
    L_with_diag_indices = L_indices.copy()
    L_with_diag_indptr = L_indptr.copy()
    for i in range(n):
        pos = L_with_diag_indptr[i]
        L_with_diag_data.insert(pos, 1.0)
        L_with_diag_indices.insert(pos, i)
        for j in range(i + 1, n + 1):
            L_with_diag_indptr[j] += 1
    current_ptr = 0
    for j in range(n):
        U_indptr[j] = current_ptr
        for row_idx, val in U_temp[j]:
            if row_idx >= j:
                U_data.append(val)
                U_indices.append(row_idx)
                current_ptr += 1
    U_indptr[n] = current_ptr
    L = CSCMatrix(L_with_diag_data, L_with_diag_indices, L_with_diag_indptr, (n, n))
    U = CSCMatrix(U_data, U_indices, U_indptr, (n, n))
    return (L, U)

def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None
    L, U = lu_result
    n = len(b)
    y = b.copy()
    for i in range(n):
        for j in range(i):
            found = False
            for k in range(L.indptr[j], L.indptr[j + 1]):
                if L.indices[k] == i:
                    y[i] -= L.data[k] * y[j]
                    found = True
                    break
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_ux = 0.0
        for j in range(i + 1, n):
            for k in range(U.indptr[j], U.indptr[j + 1]):
                if U.indices[k] == i:
                    sum_ux += U.data[k] * x[j]
                    break
        u_ii = 0.0
        found = False
        for k in range(U.indptr[i], U.indptr[i + 1]):
            if U.indices[k] == i:
                u_ii = U.data[k]
                found = True
                break
        if not found or abs(u_ii) < 1e-12:
            return None
        x[i] = (y[i] - sum_ux) / u_ii
    return x

def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None
    _, U = lu_result
    n = U.shape[0]
    det = 1.0
    for i in range(n):
        found = False
        for k in range(U.indptr[i], U.indptr[i + 1]):
            if U.indices[k] == i:
                det *= U.data[k]
                found = True
                break
        if not found:
            return 0.0
    return det

