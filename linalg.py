from CSR import CSRMatrix
from mytypes import Vector
from typing import Tuple, Optional
from collections import defaultdict

THRESHOLD = 1e-10

def lu_decomposition(A: CSRMatrix) -> Optional[Tuple[CSRMatrix, CSRMatrix]]:
    """
    LU-разложение для CSR матрицы с частичным выбором главного элемента.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    L с единицами на диагонали.
    """
    n = A.shape[0]

    U_data = A.data[:]
    U_indices = A.indices[:]
    U_indptr = A.indptr[:]
    U = CSRMatrix(U_data, U_indices, U_indptr, (n, n))

    L_data = [1.0] * n
    L_indices = list(range(n))
    L_indptr = list(range(n + 1))
    L = CSRMatrix(L_data, L_indices, L_indptr, (n, n))
    
    def get_row_dict(matrix, i):
        """Получить строку i как словарь {столбец: значение}."""
        start = matrix.indptr[i]
        end = matrix.indptr[i + 1]
        row_dict = {}
        for idx in range(start, end):
            j = matrix.indices[idx]
            row_dict[j] = matrix.data[idx]
        return row_dict
    
    def set_row_from_dict(matrix, i, row_dict):
        """Установить строку i из словаря."""
        all_rows = []
        for row_idx in range(n):
            if row_idx == i:
                sorted_items = sorted((j, val) for j, val in row_dict.items() if abs(val) > THRESHOLD)
                all_rows.append(sorted_items)
            else:
                start = matrix.indptr[row_idx]
                end = matrix.indptr[row_idx + 1]
                row_items = [(matrix.indices[idx], matrix.data[idx]) for idx in range(start, end)]
                all_rows.append(row_items)

        new_data, new_indices, new_indptr = [], [], [0]
        for row_items in all_rows:
            for j, val in row_items:
                new_data.append(val)
                new_indices.append(j)
            new_indptr.append(len(new_data))
        
        matrix.data = new_data
        matrix.indices = new_indices
        matrix.indptr = new_indptr
    
    for k in range(n - 1):
        row_k = get_row_dict(U, k)
        u_kk = row_k.get(k, 0.0)

        if abs(u_kk) < THRESHOLD:
            pivot_row = k
            max_val = 0.0
            
            for i in range(k + 1, n):
                row_i = get_row_dict(U, i)
                val = abs(row_i.get(k, 0.0))
                if val > max_val:
                    max_val = val
                    pivot_row = i
            
            if max_val < THRESHOLD:
                return None

            row_k, row_pivot = get_row_dict(U, k), get_row_dict(U, pivot_row)
            set_row_from_dict(U, k, row_pivot)
            set_row_from_dict(U, pivot_row, row_k)

            for j in range(k):
                l_kj = L.get_element(k, j)
                l_pj = L.get_element(pivot_row, j)

                row_k_l = get_row_dict(L, k)
                row_p_l = get_row_dict(L, pivot_row)
                
                if abs(l_pj) > THRESHOLD:
                    row_k_l[j] = l_pj
                elif j in row_k_l:
                    del row_k_l[j]
                
                if abs(l_kj) > THRESHOLD:
                    row_p_l[j] = l_kj
                elif j in row_p_l:
                    del row_p_l[j]
                
                set_row_from_dict(L, k, row_k_l)
                set_row_from_dict(L, pivot_row, row_p_l)

            u_kk = get_row_dict(U, k).get(k, 0.0)

        for i in range(k + 1, n):
            row_i = get_row_dict(U, i)
            u_ik = row_i.get(k, 0.0)
            
            if abs(u_ik) > THRESHOLD:
                factor = u_ik / u_kk
                
                row_i_l = get_row_dict(L, i)
                row_i_l[k] = factor
                set_row_from_dict(L, i, row_i_l)

                row_k = get_row_dict(U, k)
                new_row_i = {}

                for j, val in row_i.items():
                    if j != k:
                        new_row_i[j] = val

                for j, val_k in row_k.items():
                    if j > k:
                        current = new_row_i.get(j, 0.0)
                        new_val = current - factor * val_k
                        if abs(new_val) > THRESHOLD:
                            new_row_i[j] = new_val
                        elif j in new_row_i:
                            del new_row_i[j]
                
                set_row_from_dict(U, i, new_row_i)
    
    return L, U

def solve_SLAE_lu(A: CSRMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    lu_result = lu_decomposition(A)

    if lu_result is None:
        return None
    
    L, U = lu_result
    n = A.shape[0]

    y = [0.0] * n
    for i in range(n):
        sum_val = 0.0
        start = L.indptr[i]
        end = L.indptr[i + 1]
        for idx in range(start, end):
            j = L.indices[idx]
            if j < i:
                sum_val += L.data[idx] * y[j]
        y[i] = b[i] - sum_val

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_val = 0.0
        start = U.indptr[i]
        end = U.indptr[i + 1]

        diag = 0.0
        for idx in range(start, end):
            j = U.indices[idx]
            if j == i:
                diag = U.data[idx]
            elif j > i:
                sum_val += U.data[idx] * x[j]
        
        if abs(diag) < THRESHOLD:
            return None
        
        x[i] = (y[i] - sum_val) / diag
    
    return x

def find_det_with_lu(A: CSRMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    """
    lu_result = lu_decomposition(A)

    if lu_result is None:
        return None
    
    _, U = lu_result
    n = A.shape[0]
    
    det = 1.0
    for i in range(n):
        start = U.indptr[i]
        end = U.indptr[i + 1]
        diag = 0.0
        for idx in range(start, end):
            if U.indices[idx] == i:
                diag = U.data[idx]
                break
        det *= diag
    
    return det