from CSC import CSCMatrix
from type import Vector
from typing import Tuple, Optional, List, Dict

EPSILON = 1e-12


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix, List[int]]]:
    n = A.shape[0]
    
    if n != A.shape[1]:
        raise ValueError("LU-разложение требует квадратную матрицу")
    
    # Создаем рабочие структуры
    L_rows = [{} for _ in range(n)]
    U_rows = [{} for _ in range(n)]
    P = list(range(n))
    
    # Копируем A в U
    for j in range(n):
        for idx in range(A.indptr[j], A.indptr[j + 1]):
            i = A.indices[idx]
            U_rows[i][j] = float(A.data[idx])
    
    for k in range(n):
        # Частичный выбор
        max_val = 0.0
        max_row = k
        
        for i in range(k, n):
            if k in U_rows[i]:
                val = abs(U_rows[i][k])
                if val > max_val:
                    max_val = val
                    max_row = i
        
        if max_val < EPSILON:
            return None
        
        # Перестановка строк
        if max_row != k:
            P[k], P[max_row] = P[max_row], P[k]
            U_rows[k], U_rows[max_row] = U_rows[max_row], U_rows[k]
            L_rows[k], L_rows[max_row] = L_rows[max_row], L_rows[k]
        
        u_kk = U_rows[k].get(k, 0.0)
        if abs(u_kk) < EPSILON:
            return None
        
        L_rows[k][k] = 1.0
        
        # Обновление строк
        for i in range(k + 1, n):
            if k in U_rows[i]:
                l_ik = U_rows[i][k] / u_kk
                L_rows[i][k] = l_ik
                
                for j, u_kj in U_rows[k].items():
                    if j >= k:
                        new_val = U_rows[i].get(j, 0.0) - l_ik * u_kj
                        if abs(new_val) > EPSILON:
                            U_rows[i][j] = new_val
                        elif j in U_rows[i]:
                            del U_rows[i][j]
                
                del U_rows[i][k]
    
    # Конвертация в CSC
    L_data, L_indices, L_indptr = [], [], [0]
    U_data, U_indices, U_indptr = [], [], [0]
    
    # L в CSC
    L_cols = [{} for _ in range(n)]
    for i in range(n):
        for j, val in L_rows[i].items():
            if i >= j:
                L_cols[j][i] = val
    
    for j in range(n):
        rows = sorted(L_cols[j].keys())
        for i in rows:
            L_data.append(L_cols[j][i])
            L_indices.append(i)
        L_indptr.append(len(L_data))
    
    # U в CSC
    U_cols = [{} for _ in range(n)]
    for i in range(n):
        for j, val in U_rows[i].items():
            if i <= j:
                U_cols[j][i] = val
    
    for j in range(n):
        rows = sorted(U_cols[j].keys())
        for i in rows:
            U_data.append(U_cols[j][i])
            U_indices.append(i)
        U_indptr.append(len(U_data))
    
    return (CSCMatrix(L_data, L_indices, L_indptr, (n, n)),
            CSCMatrix(U_data, U_indices, U_indptr, (n, n)), P)


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    lu_result = lu_decomposition(A)
    
    if lu_result is None:
        return None
    
    L, U, P = lu_result
    n = A.shape[0]
    
    # Применяем перестановку
    b_perm = [b[P[i]] for i in range(n)]
    
    # Ly = b_perm
    y = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L.get(i, j) * y[j]
        y[i] = b_perm[i] - s
    
    # Ux = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += U.get(i, j) * x[j]
        
        diag = U.get(i, i)
        if abs(diag) < EPSILON:
            return None
        
        x[i] = (y[i] - s) / diag
    
    return x


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    lu_result = lu_decomposition(A)
    
    if lu_result is None:
        return None
    
    _, U, P = lu_result
    n = A.shape[0]
    
    # Подсчет перестановок
    visited = [False] * n
    swaps = 0
    
    for i in range(n):
        if not visited[i]:
            cycle_len = 0
            j = i
            while not visited[j]:
                visited[j] = True
                j = P[j]
                cycle_len += 1
            if cycle_len > 1:
                swaps += cycle_len - 1
    
    sign = 1 if swaps % 2 == 0 else -1
    
    # Определитель
    det = sign
    for i in range(n):
        diag = U.get(i, i)
        if abs(diag) < EPSILON:
            return 0.0
        det *= diag
    
    return det