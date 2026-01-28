import heapq
from CSC import CSCMatrix
from CSR import CSRMatrix
from .type import Vector
from typing import Tuple, Optional

def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    Возвращает (L,U) нижнюю и верхнюю треугольные матрицы
    Матрица L будет хранить единицы на своей главной диагонали
    """
    pass
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        return None
    L_cache_idx = [None] * n
    L_cache_val = [None] * n
    L_data_final = []
    L_indices_final = []
    L_indptr_final = [0]
    U_data_final = []
    U_indices_final = []
    U_indptr_final = [0]
    x = [0] * n
    visited = [False] * n
    for j in range(n):
        col_start = A.indptr[j]
        col_end = A.indptr[j + 1]
        touched = []
        heap = []
        for k in range(col_start, col_end):
            r = A.indices[k]
            val = A.data[k]
            x[r] = val
            touched.append(r)
            if r < j and not visited[r]:
                visited[r] = True
                heapq.heappush(heap, r)
        while heap:
            k = heapq.heappop(heap)
            pivot_val = x[k]
            if pivot_val == 0:
                continue
            l_idx = L_cache_idx[k]
            l_val = L_cache_val[k]
            if not l_idx:
                continue
            for r, v in zip(l_idx, l_val):
                if x[r] == 0:
                    touched.append(r)
                    if r < j and not visited[r]:
                        visited[r] = True
                        heapq.heappush(heap, r)
                x[r] -= v * pivot_val
        diag = x[j]
        if abs(diag) < 1e-15:
            return None
        touched.sort()
        curr_l_idx = []
        curr_l_val = []
        curr_u_idx = []
        curr_u_val = []
        for r in touched:
            val = x[r]
            x[r] = 0
            visited[r] = False
            if abs(val) < 1e-15:
                continue
            if r <= j:
                curr_u_idx.append(r)
                curr_u_val.append(val)
            else:
                val /= diag
                curr_l_idx.append(r)
                curr_l_val.append(val)
        L_cache_idx[j] = curr_l_idx
        L_cache_val[j] = curr_l_val
        U_indices_final.extend(curr_u_idx)
        U_data_final.extend(curr_u_val)
        U_indptr_final.append(len(U_data_final))
        L_indices_final.append(j)
        L_data_final.append(1)
        L_indices_final.extend(curr_l_idx)
        L_data_final.extend(curr_l_val)
        L_indptr_final.append(len(L_data_final))
    L = CSCMatrix(L_data_final, L_indices_final, L_indptr_final, (n, n))
    U = CSCMatrix(U_data_final, U_indices_final, U_indptr_final, (n, n))
    return L, U


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение
    """
    pass
    if A.shape[0] != len(b):
        return None
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None
    L, U = lu_result
    n = len(b)
    x = list(b)
    for j in range(n):
        xj = x[j]
        start = L.indptr[j]
        end = L.indptr[j + 1]
        for k in range(start + 1, end):
            row = L.indices[k]
            val = L.data[k]
            x[row] -= val * xj
    for j in range(n - 1, -1, -1):
        start = U.indptr[j]
        end = U.indptr[j + 1]
        if start == end:
            return None
        diag_idx = end - 1
        diag_row = U.indices[diag_idx]
        diag_val = U.data[diag_idx]
        if diag_row != j or diag_val == 0:
            return None
        x[j] /= diag_val
        xj = x[j]
        for k in range(start, end - 1):
            row = U.indices[k]
            val = U.data[k]
            x[row] -= val * xj
    return x


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Определитель через LU-разложение
    """
    pass
    if A.shape[0] != A.shape[1]:
        return None
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return 0
    L, U = lu_result
    n = A.shape[0]
    det = 1
    for j in range(n):
        col_start = U.indptr[j]
        col_end = U.indptr[j + 1]
        if col_start == col_end:
            return 0
        last_idx = col_end - 1
        diag_row = U.indices[last_idx]
        diag_val = U.data[last_idx]
        if diag_row != j:
            return 0
        det *= diag_val
        if det == 0:
            return 0
    return det