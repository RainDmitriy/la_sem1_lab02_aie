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
    n = A.shape[0]
    if n == 0 or len(A.to_dense()[0]) != n:
        return None
    rows = []
    for i in range(n):
        row_dict = {}
        for col in range(n):
            start_idx = A.indptr[col]
            end_idx = A.indptr[col + 1]
            for idx in range(start_idx, end_idx):
                if A.indices[idx] == i:
                    row_dict[col] = A.data[idx]
                    break
        rows.append(row_dict)
    L_rows = [{} for _ in range(n)]
    U_rows = [{} for _ in range(n)]
    for i in range(n):
        for j, val in rows[i].items():
            if i <= j:
                U_rows[i][j] = float(val)
    for k in range(n):
        if k not in U_rows[k]:
            found = False
            for i in range(k + 1, n):
                if k in U_rows[i]:
                    U_rows[k], U_rows[i] = U_rows[i], U_rows[k]
                    L_rows[k], L_rows[i] = L_rows[i], L_rows[k]
                    rows[k], rows[i] = rows[i], rows[k]
                    found = True
                    break
            if not found:
                return None
        u_kk = U_rows[k].get(k, 0.0)
        if abs(u_kk) < 1e-12:
            return None
        for i in range(k + 1, n):
            if k in U_rows[i]:
                L_rows[i][k] = U_rows[i][k] / u_kk
                if k in U_rows[i]:
                    del U_rows[i][k]
            for j in range(k + 1, n):
                if k in U_rows[k] and j in U_rows[k]:
                    if j in U_rows[i]:
                        U_rows[i][j] = U_rows[i].get(j, 0.0) - L_rows[i].get(k, 0.0) * U_rows[k][j]
                    elif L_rows[i].get(k, 0.0) != 0:
                        U_rows[i][j] = -L_rows[i].get(k, 0.0) * U_rows[k][j]
        L_rows[k][k] = 1.0
    L_data, L_indices, L_indptr = [], [], [0]
    U_data, U_indices, U_indptr = [], [], [0]
    for col in range(n):
        current_count = 0
        for row in range(n):
            if col in L_rows[row]:
                val = L_rows[row][col]
                if abs(val) > 1e-12 or row == col:
                    L_data.append(val)
                    L_indices.append(row)
                    current_count += 1
        L_indptr.append(L_indptr[-1] + current_count)
    for col in range(n):
        current_count = 0
        for row in range(n):
            if col in U_rows[row]:
                val = U_rows[row][col]
                if abs(val) > 1e-12:
                    U_data.append(val)
                    U_indices.append(row)
                    current_count += 1
        U_indptr.append(U_indptr[-1] + current_count)
    L = CSCMatrix(L_data, L_indices, L_indptr, (n, n))
    U = CSCMatrix(U_data, U_indices, U_indptr, (n, n))
    return L, U

def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None
    L, U = lu_result
    n = len(b)
    if n != A.shape[0]:
        return None
    L_dense = L.to_dense()
    U_dense = U.to_dense()
    y = [0.0] * n
    for i in range(n):
        sum_val = 0.0
        for j in range(i):
            sum_val += L_dense[i][j] * y[j]
        y[i] = b[i] - sum_val
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_val = 0.0
        for j in range(i + 1, n):
            sum_val += U_dense[i][j] * x[j]
        if abs(U_dense[i][i]) < 1e-12:
            return None
        x[i] = (y[i] - sum_val) / U_dense[i][i]
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
    U_dense = U.to_dense()
    n = len(U_dense)
    det = 1.0
    for i in range(n):
        det *= U_dense[i][i]
    return det

