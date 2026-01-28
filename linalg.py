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
    
    # SPA (Sparse Accumulator) - плотный вектор
    x = [0.0] * n
    # Кэш чистого вектора для моментальной очистки
    zeros = [0.0] * n
    
    # Кэш столбцов L для ускорения lookup (храним (indices, values))
    L_cols_cache = [] 

    for j in range(n):
        # 1. Загрузка столбца A в аккумулятор
        # range гарантированно конечен
        start_ptr = A.indptr[j]
        end_ptr = A.indptr[j+1]
        
        for k in range(start_ptr, end_ptr):
            x[A.indices[k]] += A.data[k]

        # 2. Исключение (Elimination)
        # Цикл строго от 0 до j-1
        for k in range(j):
            ukj = x[k]
            if ukj == 0:
                continue
            
            # Достаем закэшированный столбец
            l_rows, l_vals = L_cols_cache[k]
            
            # Проход по списку конечной длины
            for r, val in zip(l_rows, l_vals):
                x[r] -= val * ukj

        # 3. Формирование столбца U
        u_col_data = []
        u_col_ind = []
        
        # Сбор данных U (до диагонали)
        for i in range(j + 1):
            val = x[i]
            if val != 0:
                u_col_data.append(val)
                u_col_ind.append(i)
        
        U_data.extend(u_col_data)
        U_indices.extend(u_col_ind)
        U_indptr.append(len(U_data))

        # 4. Пивот (диагональный элемент)
        ujj = x[j]
        if abs(ujj) < 1e-15:
            return None # Выход, если матрица вырождена

        # 5. Формирование столбца L
        l_col_data_cache = []
        l_col_ind_cache = []
        
        L_data.append(1.0) # Единица на диагонали L
        L_indices.append(j)
        
        # Сбор данных L (после диагонали)
        for i in range(j + 1, n):
            val = x[i]
            if val != 0:
                val /= ujj
                l_col_data_cache.append(val)
                l_col_ind_cache.append(i)
                L_data.append(val)
                L_indices.append(i)
        
        L_indptr.append(len(L_data))
        
        # Сохраняем "хвост" для будущих итераций
        L_cols_cache.append((l_col_ind_cache, l_col_data_cache))

        # 6. Моментальная очистка вектора
        # Это не цикл Python, это низкоуровневая операция C, работает мгновенно
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
    
    # Прямой ход
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
                y[row] -= L.data[k] * yj

    # Обратный ход
    x = y
    for j in range(n - 1, -1, -1):
        start = U.indptr[j]
        end = U.indptr[j+1]
        ujj = 0.0
        
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
                x[row] -= U.data[k] * xj
                
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