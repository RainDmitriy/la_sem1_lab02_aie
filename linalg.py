from CSC import CSCMatrix
from CSR import CSRMatrix
from matrix_types import Vector
from typing import Tuple, Optional

def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    n = A.shape[0]
    if n != A.shape[1]:
        raise ValueError("Matrix must be square")

    # Локализация списков для ускорения доступа
    A_data = A.data
    A_indices = A.indices
    A_indptr = A.indptr

    L_data, L_indices, L_indptr = [], [], [0]
    U_data, U_indices, U_indptr = [], [], [0]
    
    # SPA (Sparse Accumulator)
    x = [0.0] * n
    # Кэш нулей
    zeros = [0.0] * n
    
    # Кэш столбцов L: хранит (indices_list, values_list)
    L_cols_cache = [] 

    for j in range(n):
        start_ptr = A_indptr[j]
        end_ptr = A_indptr[j+1]

        if start_ptr < end_ptr:
            first_row = A_indices[start_ptr]
        else:
            first_row = j  # Столбец пуст (или только диагональ будет заполнена позже)

        # 1. Загрузка столбца A (Scatter)
        for k in range(start_ptr, end_ptr):
            x[A_indices[k]] += A_data[k]

        # 2. Исключение (Elimination)
        # Начинаем цикл с first_row вместо 0! Огромное ускорение для ленточных матриц.
        for k in range(first_row, j):
            ukj = x[k]
            if ukj == 0:
                continue
            
            # Достаем закэшированный столбец L
            l_rows, l_vals = L_cols_cache[k]
            
            # Inner loop
            for r, val in zip(l_rows, l_vals):
                x[r] -= val * ukj

        # 3. Формирование U (Gather U)
        u_col_data = []
        u_col_ind = []
        
        # Сбор начинаем тоже с first_row, раньше там нули
        for i in range(first_row, j + 1):
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

        # 5. Формирование L (Gather L)
        l_col_data_cache = []
        l_col_ind_cache = []
        
        L_data.append(1.0)
        L_indices.append(j)
        
        # Хвост L собираем
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

        # 6. Очистка
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
    
    # Локализация для скорости
    L_data, L_indices, L_indptr = L.data, L.indices, L.indptr
    U_data, U_indices, U_indptr = U.data, U.indices, U.indptr
    
    y = list(b)
    
    # Прямой ход (L y = b)
    for j in range(n):
        yj = y[j]
        if yj == 0:
            continue
        
        start = L_indptr[j]
        end = L_indptr[j+1]
        # В L диагональ - первый элемент или не хранится (тут хранится 1.0)
        # Нам нужны элементы строго ниже диагонали
        for k in range(start, end):
            row = L_indices[k]
            if row > j:
                y[row] -= L_data[k] * yj

    # Обратный ход (U x = y)
    x = y
    for j in range(n - 1, -1, -1):
        start = U_indptr[j]
        end = U_indptr[j+1]
        
        # Поиск диагонали в U.
        # Т.к. U верхнетреугольная CSC, диагональный элемент - последний в столбце.
        # (если сортировка соблюдена). Проверим это предположение.
        # Если сортировка есть, indices[end-1] == j.
        # Это короткий цикл.
        
        ujj = 0.0
        diag_idx = -1
        
        # Оптимизация: идем с конца, диагональ обычно внизу столбца U
        for k in range(end - 1, start - 1, -1):
            if U_indices[k] == j:
                ujj = U_data[k]
                break
        
        if ujj == 0:
            return None
            
        x[j] /= ujj
        xj = x[j]
        
        # Вычитаем из верхних строк
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
        # Диагональ в U обычно последняя
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