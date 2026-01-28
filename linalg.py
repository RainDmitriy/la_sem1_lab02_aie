from CSC import CSCMatrix
from CSR import CSRMatrix
from matrix_types import Vector
from typing import Tuple, Optional


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    n = A.shape[0]
    if n != A.shape[1]:
        raise ValueError("Matrix must be square")

    L_data, L_indices, L_indptr = [], [], [0]
    U_data, U_indices, U_indptr = [], [], [0]
    
    # SPA (Sparse Accumulator)
    x = [0.0] * n
    # Кэш нулей для быстрого сброса
    zeros = [0.0] * n
    
    L_cols_cache = [] 

    for j in range(n):
        # 1. Загрузка столбца A
        start_ptr = A.indptr[j]
        end_ptr = A.indptr[j+1]
        
        for k in range(start_ptr, end_ptr):
            x[A.indices[k]] += A.data[k]

        # 2. Исключение (Самый горячий цикл)
        for k in range(j):
            ukj = x[k]
            if ukj == 0:
                continue
            
            # Разворачиваем данные из кэша
            l_rows, l_vals = L_cols_cache[k]
            
            # В Python zip в цикле может быть медленным, 
            # но это быстрее, чем range по индексам списка
            for r, val in zip(l_rows, l_vals):
                x[r] -= val * ukj

        # 3. Формирование U
        u_col_data = []
        u_col_ind = []
        # Проходим только до j, собирая ненулевые
        for i in range(j + 1):
            val = x[i]
            if val != 0:
                u_col_data.append(val)
                u_col_ind.append(i)
        
        U_data.extend(u_col_data)
        U_indices.extend(u_col_ind)
        U_indptr.append(len(U_data))

        # 4. Пивот
        ujj = x[j]
        if abs(ujj) < 1e-15:
            return None

        # 5. Формирование L
        l_col_data_cache = []
        l_col_ind_cache = []
        
        L_data.append(1.0)
        L_indices.append(j)
        
        # Нормализация хвоста
        for i in range(j + 1, n):
            val = x[i]
            if val != 0:
                val /= ujj
                l_col_data_cache.append(val)
                l_col_ind_cache.append(i)
                L_data.append(val)
                L_indices.append(i)
        
        L_indptr.append(len(L_data))
        L_cols_cache.append((l_col_ind_cache, l_col_data_cache))

        # 6. Быстрая очистка вектора срезом (Slice assignment)
        # Это работает быстрее, чем loop по touched_indices для плотных векторов < 1000 элементов
        x[:] = zeros

    return (
        CSCMatrix(L_data, L_indices, L_indptr, A.shape),
        CSCMatrix(U_data, U_indices, U_indptr, A.shape)
    )

def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    lu = lu_decomposition(A)
    if lu is None:
        return None
    L, U = lu
    n = A.shape[0]
    
    y = list(b)
    for j in range(n):
        yj = y[j]
        if yj == 0:
            continue
        start = L.indptr[j]
        end = L.indptr[j+1]
        for k in range(start, end):
            row = L.indices[k]
            if row > j:
                val = L.data[k]
                y[row] -= val * yj

    x = y
    for j in range(n - 1, -1, -1):
        start = U.indptr[j]
        end = U.indptr[j+1]
        ujj = 0.0
        
        # Поиск диагонального элемента
        for k in range(start, end):
            if U.indices[k] == j:
                ujj = U.data[k]
                break
        
        if ujj == 0:
            return None
            
        x[j] /= ujj
        xj = x[j]
        
        for k in range(start, end):
            row = U.indices[k]
            if row < j:
                val = U.data[k]
                x[row] -= val * xj
                
    return x

def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    lu = lu_decomposition(A)
    if lu is None:
        return 0.0
        
    _, U = lu
    n = U.shape[0]
    det = 1.0
    
    for j in range(n):
        start = U.indptr[j]
        end = U.indptr[j+1]
        diag_found = False
        for k in range(start, end):
            if U.indices[k] == j:
                det *= U.data[k]
                diag_found = True
                break
        if not diag_found:
            return 0.0
            
    return det
