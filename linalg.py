from CSC import CSCMatrix
from CSR import CSRMatrix
from matrix_types import Vector
from typing import Tuple, Optional

def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    n = A.shape[0]
    if n != A.shape[1]:
        raise ValueError("Matrix must be square")

    # Локализация
    A_data = A.data
    A_indices = A.indices
    A_indptr = A.indptr

    L_data, L_indices, L_indptr = [], [], [0]
    U_data, U_indices, U_indptr = [], [], [0]
    
    # Плотный вектор
    x = [0.0] * n
    
    # Кэш столбцов L
    L_cols_cache = [] 

    for j in range(n):
        start_ptr = A_indptr[j]
        end_ptr = A_indptr[j+1]
        
        # Список индексов, которые нужно обнулить в конце шага
        touched = []
        
        # Skyline Optimization
        first_row = 0
        if start_ptr < end_ptr:
            first_row = A_indices[start_ptr]
        else:
            first_row = j 

        # 1. Scatter
        for k in range(start_ptr, end_ptr):
            r = A_indices[k]
            if x[r] == 0:
                touched.append(r)
            x[r] += A_data[k]

        # 2. Elimination
        for k in range(first_row, j):
            ukj = x[k]
            if ukj == 0:
                continue
            
            l_rows, l_vals = L_cols_cache[k]
            
            for r, val in zip(l_rows, l_vals):
                if x[r] == 0:
                    touched.append(r)
                x[r] -= val * ukj

        # 3. Gather U
        u_col_data = []
        u_col_ind = []
        
        # Сбор U (идем по touched, но нужно в порядке индексов)
        # Для скорости идем циклом, так как диапазон ограничен Skyline
        # Если диапазон огромный, можно фильтровать touched, но sorted(touched) тоже накладно.
        # Обычно Skyline дает узкий диапазон.
        for i in range(first_row, j + 1):
            val = x[i]
            if val != 0:
                u_col_data.append(val)
                u_col_ind.append(i)
        
        U_data.extend(u_col_data)
        U_indices.extend(u_col_ind)
        U_indptr.append(len(U_data))

        # 4. Pivot
        ujj = x[j]
        if abs(ujj) < 1e-15:
            return None 

        # 5. Gather L
        l_col_data_cache = []
        l_col_ind_cache = []
        
        L_data.append(1.0)
        L_indices.append(j)
        
        # Хвост L
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

        # 6. Sparse Clean
        for idx in touched:
            x[idx] = 0.0

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
    
    L_data, L_indices, L_indptr = L.data, L.indices, L.indptr
    U_data, U_indices, U_indptr = U.data, U.indices, U.indptr
    
    y = list(b)
    
    # L y = b
    for j in range(n):
        yj = y[j]
        if yj == 0:
            continue
        
        start = L_indptr[j]
        end = L_indptr[j+1]
        for k in range(start, end):
            row = L_indices[k]
            if row > j:
                y[row] -= L_data[k] * yj

    # U x = y
    x = y
    for j in range(n - 1, -1, -1):
        start = U_indptr[j]
        end = U_indptr[j+1]
        
        ujj = 0.0
        # Ищем диагональ с конца (обычно она там)
        for k in range(end - 1, start - 1, -1):
            if U_indices[k] == j:
                ujj = U_data[k]
                break
        
        if ujj == 0:
            return None
            
        x[j] /= ujj
        xj = x[j]
        
        for k in range(start, end):
            row = U_indices[k]
            if row < j:
                x[row] -= U_data[k] * xj
                
    return x

def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    lu = lu_decomposition(A)
    if lu is None:
        return 0.0
        
    _, U = lu
    n = U.shape[0]
    det = 1.0
    
    U_data, U_indices, U_indptr = U.data, U.indices, U.indptr
    
    for j in range(n):
        start = U_indptr[j]
        end = U_indptr[j+1]
        diag_found = False
        if end > start and U_indices[end-1] == j:
             det *= U_data[end-1]
             diag_found = True
        else:
            for k in range(start, end):
                if U_indices[k] == j:
                    det *= U_data[k]
                    diag_found = True
                    break
        
        if not diag_found:
            return 0.0
            
    return det