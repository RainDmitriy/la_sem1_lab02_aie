from CSR import CSRMatrix
from mytypes import Vector
from typing import Optional, Tuple, List

def lu_decomposition(A: CSRMatrix) -> Optional[Tuple[CSRMatrix, CSRMatrix]]:
    """
    LU-разложение для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    Ожидается, что матрица L хранит единицы на главной диагонали.
    """
    if not isinstance(A, CSRMatrix):
        A = A._to_csr()

    n = A.shape[0]

    U_data = A.data[:]
    U_indices = A.indices[:]
    U_indptr = A.indptr[:]
    U = CSRMatrix(U_data, U_indices, U_indptr, (n, n))

    L_data, L_indices, L_indptr = [], [], [0] * (n + 1)
    
    for i in range(n):
        L_data.append(1.0)
        L_indices.append(i)
        L_indptr[i + 1] = i + 1
    
    L = CSRMatrix(L_data, L_indices, L_indptr, (n, n))

    for k in range(n - 1):
        row_k = U.get_row(k)
        u_kk = 0.0
        for col, val in row_k:
            if col == k:
                u_kk = val
                break
        if abs(u_kk) < 1e-10:
            pivot_found = False
            
            for i in range(k + 1, n):
                row_i = U.get_row(i)
                for col, val in row_i:
                    if col == k and abs(val) > 1e-10:
                        U.swap_rows(k, i)
                        L.swap_rows(k, i)
                        if k > 0:
                            for j in range(k):
                                l_kj = L.get_element(k, j)
                                l_ij = L.get_element(i, j)
                                
                                row_k_l = L.get_row(k)
                                row_i_l = L.get_row(i)
                                
                                new_row_k = [(c, v) for c, v in row_k_l if c != j]
                                new_row_i = [(c, v) for c, v in row_i_l if c != j]
                                
                                if abs(l_ij) > 1e-12:
                                    new_row_k.append((j, l_ij))
                                if abs(l_kj) > 1e-12:
                                    new_row_i.append((j, l_kj))
                                
                                L.set_row(k, sorted(new_row_k, key=lambda x: x[0]))
                                L.set_row(i, sorted(new_row_i, key=lambda x: x[0]))
                        
                        pivot_found = True
                        break
                
                if pivot_found:
                    break
            
            if not pivot_found:
                return None

            row_k = U.get_row(k)
            u_kk = 0.0
            for col, val in row_k:
                if col == k:
                    u_kk = val
                    break

        for i in range(k + 1, n):
            row_i = U.get_row(i)
            
            u_ik = 0.0
            for col, val in row_i:
                if col == k:
                    u_ik = val
                    break
            
            if abs(u_ik) > 1e-12:
                factor = u_ik / u_kk

                l_row = L.get_row(i)
                new_l_row = [(c, v) for c, v in l_row if c != k]
                new_l_row.append((k, factor))
                L.set_row(i, sorted(new_l_row, key=lambda x: x[0]))

                new_row_i = []
                p1, p2 = 0, 0
                len1, len2 = len(row_i), len(row_k)
                
                while p1 < len1 and p2 < len2:
                    j1, val1 = row_i[p1]
                    j2, val2 = row_k[p2]
                    
                    if j1 < j2:
                        if j1 != k:
                            new_row_i.append((j1, val1))
                        p1 += 1
                    elif j1 > j2:
                        if j2 >= k:
                            new_val = -factor * val2
                            if abs(new_val) > 1e-12:
                                new_row_i.append((j2, new_val))
                        p2 += 1
                    else:
                        if j1 == k:
                            new_val = 0.0
                        else:
                            new_val = val1 - factor * val2
                        
                        if abs(new_val) > 1e-12:
                            new_row_i.append((j1, new_val))
                        p1 += 1
                        p2 += 1

                while p1 < len1:
                    j1, val1 = row_i[p1]
                    if j1 != k:
                        new_row_i.append((j1, val1))
                    p1 += 1
                
                while p2 < len2:
                    j2, val2 = row_k[p2]
                    if j2 > k:
                        new_val = -factor * val2
                        if abs(new_val) > 1e-12:
                            new_row_i.append((j2, new_val))
                    p2 += 1
                
                U.set_row(i, sorted(new_row_i, key=lambda x: x[0]))
    
    return L, U

def solve_SLAE_lu(A: CSRMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    if len(b) != A.shape[0]:
        raise ValueError(f"Размер вектора b ({len(b)}) не равен размеру матрицы A ({A.shape[0]})")
    
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None
    
    L, U = lu_result
    n = len(b)

    y = [0.0] * n
    for i in range(n):
        sum_val = 0.0
        row_L = L.get_row(i)
        for col, val in row_L:
            if col < i:
                sum_val += val * y[col]
        y[i] = b[i] - sum_val

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_val = 0.0
        row_U = U.get_row(i)
        for col, val in row_U:
            if col > i:
                sum_val += val * x[col]
        
        diag = 0.0
        for col, val in row_U:
            if col == i:
                diag = val
                break
        
        if abs(diag) < 1e-12:
            return None
        
        x[i] = (y[i] - sum_val) / diag
    
    return x

def find_det_with_lu(A: CSRMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None
    
    _, U = lu_result
    n = A.shape[0]
    
    det = 1.0
    for i in range(n):
        diag = U.get_element(i, i)
        det *= diag
    
    return det