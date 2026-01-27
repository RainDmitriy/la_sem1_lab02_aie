from CSC import CSCMatrix
from mytypes import Vector
from typing import Tuple, Optional, Dict, List
from collections import defaultdict

THRESHOLD = 1e-10

def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы.
    Возвращает (L, U) в CSC формате.
    L с единицами на диагонали.
    """
    n = A.shape[0]
    
    U_coo = A._to_coo()
    U_data = U_coo.data.copy()
    U_rows = U_coo.row.copy()
    U_cols = U_coo.col.copy()

    L_data = []
    L_rows = []
    L_cols = []

    for i in range(n):
        L_data.append(1.0)
        L_rows.append(i)
        L_cols.append(i)
    
    L_coo = type(U_coo)(L_data, L_rows, L_cols, (n, n))

    U_dict = defaultdict(lambda: defaultdict(float))
    for idx in range(len(U_data)):
        i = U_rows[idx]
        j = U_cols[idx]
        U_dict[i][j] = U_data[idx]
    
    L_dict = defaultdict(lambda: defaultdict(float))
    for idx in range(len(L_data)):
        i = L_rows[idx]
        j = L_cols[idx]
        L_dict[i][j] = L_data[idx]
    
    for k in range(n - 1):
        pivot = U_dict[k].get(k, 0.0)

        if abs(pivot) < THRESHOLD:
            max_val = 0.0
            max_row = k
            
            for i in range(k + 1, n):
                val = abs(U_dict[i].get(k, 0.0))
                if val > max_val:
                    max_val = val
                    max_row = i
            
            if max_val < THRESHOLD:
                return None

            U_dict[k], U_dict[max_row] = U_dict[max_row], U_dict[k]

            for j in range(k):
                L_dict[k][j], L_dict[max_row][j] = L_dict[max_row][j], L_dict[k][j]

            pivot = U_dict[k].get(k, 0.0)

        for i in range(k + 1, n):
            factor = U_dict[i].get(k, 0.0) / pivot
            
            if abs(factor) > THRESHOLD:

                L_dict[i][k] = factor
                row_i = U_dict[i].copy()
                row_k = U_dict[k]

                for j in row_k:
                    if j >= k:
                        new_val = row_i.get(j, 0.0) - factor * row_k[j]
                        if abs(new_val) > THRESHOLD:
                            row_i[j] = new_val
                        elif j in row_i:
                            del row_i[j]

                if k in row_i and abs(row_i[k]) < THRESHOLD:
                    del row_i[k]
                
                U_dict[i] = row_i
    
    U_final_data = []
    U_final_rows = []
    U_final_cols = []
    
    for i in sorted(U_dict.keys()):
        for j in sorted(U_dict[i].keys()):
            val = U_dict[i][j]
            if abs(val) > THRESHOLD:
                U_final_data.append(val)
                U_final_rows.append(i)
                U_final_cols.append(j)
    
    U_final_coo = type(U_coo)(U_final_data, U_final_rows, U_final_cols, (n, n))

    L_final_data = []
    L_final_rows = []
    L_final_cols = []
    
    for i in sorted(L_dict.keys()):
        for j in sorted(L_dict[i].keys()):
            val = L_dict[i][j]
            if abs(val) > THRESHOLD:
                L_final_data.append(val)
                L_final_rows.append(i)
                L_final_cols.append(j)
    
    L_final_coo = type(U_coo)(L_final_data, L_final_rows, L_final_cols, (n, n))
    
    return L_final_coo._to_csc(), U_final_coo._to_csc()

def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решает систему линейных уравнений A*x = b через LU разложение.
    """
    if len(b) != A.shape[0]:
        raise ValueError(f"Размер вектора b ({len(b)}) не равен размеру матрицы A ({A.shape[0]})")
    
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None
    
    L, U = lu_result
    n = A.shape[0]

    y = [0.0] * n
    for i in range(n):
        sum_val = 0.0
        for j in range(i):
            found = False
            col_start = L.indptr[j]
            col_end = L.indptr[j + 1]
            for idx in range(col_start, col_end):
                if L.indices[idx] == i:
                    sum_val += L.data[idx] * y[j]
                    found = True
                    break
        y[i] = b[i] - sum_val

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_val = 0.0

        diag = 0.0
        for j in range(i + 1, n):
            col_start = U.indptr[j]
            col_end = U.indptr[j + 1]
            for idx in range(col_start, col_end):
                if U.indices[idx] == i:
                    sum_val += U.data[idx] * x[j]
                    break

        col_start = U.indptr[i]
        col_end = U.indptr[i + 1]
        for idx in range(col_start, col_end):
            if U.indices[idx] == i:
                diag = U.data[idx]
                break
        
        if abs(diag) < THRESHOLD:
            return None
        
        x[i] = (y[i] - sum_val) / diag
    
    return x

def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Вычисляет определитель матрицы через LU разложение.
    """
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None
    
    _, U = lu_result
    n = A.shape[0]
    
    det = 1.0
    for i in range(n):
        col_start = U.indptr[i]
        col_end = U.indptr[i + 1]
        diag = 0.0
        for idx in range(col_start, col_end):
            if U.indices[idx] == i:
                diag = U.data[idx]
                break
        det *= diag
    
    return det