from CSC import CSCMatrix
from CSR import CSRMatrix
from type import Vector
from typing import Tuple, Optional
import heapq


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы
    Возвращает(L, U) - нижнюю и верхнюю треугольные матрицы
    Ожидается, что матрица L хранит единицы на главной диагонали
    """
    n, m = A.shape
    if n != m:
        return None

    L_data = []
    L_indices = []
    L_indptr = [0]
    
    U_data = []
    U_indices = []
    U_indptr = [0]
    x = [0.0] * n
    marked = [False] * n

    L_cache_idx = [None] * n
    L_cache_val = [None] * n
    
    for j in range(n):
        start = A.indptr[j]
        end = A.indptr[j + 1]
        
        touched = []
        for idx in range(start, end):
            i = A.indices[idx]
            x[i] = A.data[idx]
            if not marked[i]:
                touched.append(i)
                marked[i] = True
        heap = []
        for i in touched:
            if i < j:
                heapq.heappush(heap, i)
        
        while heap:
            k = heapq.heappop(heap)
            pivot = x[k]
            
            if abs(pivot) < EPS:
                continue
            l_idx = L_cache_idx[k]
            l_val = L_cache_val[k]
            
            if not l_idx:
                continue
            for idx, val in zip(l_idx, l_val):
                i = idx
                if not marked[i]:
                    marked[i] = True
                    touched.append(i)
                    if i < j:
                        heapq.heappush(heap, i)
                
                x[i] -= val * pivot
        pivot_row = j
        pivot_val = abs(x[j])
        
        for i in range(j + 1, n):
            if abs(x[i]) > pivot_val:
                pivot_val = abs(x[i])
                pivot_row = i
        
        if pivot_val < EPS:
            return None
        if pivot_row != j:
            x[j], x[pivot_row] = x[pivot_row], x[j]
            if not marked[j]:
                marked[j] = True
                touched.append(j)
            if not marked[pivot_row]:
                marked[pivot_row] = True
                touched.append(pivot_row)
        diag = x[j]
        touched.sort()
        u_idx_curr = []
        u_val_curr = []
        l_idx_curr = []
        l_val_curr = []
        for i in touched:
            val = x[i]
            x[i] = 0.0
            marked[i] = False
            if abs(val) < EPS:
                continue
            if i <= j:
                u_idx_curr.append(i)
                u_val_curr.append(val)
            else:
                l_idx_curr.append(i)
                l_val_curr.append(val / diag)
        L_cache_idx[j] = l_idx_curr
        L_cache_val[j] = l_val_curr
        U_indices.extend(u_idx_curr)
        U_data.extend(u_val_curr)
        U_indptr.append(len(U_data))
        L_indices.append(j)
        L_data.append(1.0)
        L_indices.extend(l_idx_curr)
        L_data.extend(l_val_curr)
        L_indptr.append(len(L_data))
    
    # Создаем матрицы L и U
    L = CSCMatrix(L_data, L_indices, L_indptr, (n, n))
    U = CSCMatrix(U_data, U_indices, U_indptr, (n, n))
    
    return L, U


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение
    """
    lu = lu_decomposition(A)
    if lu is None:
        return None
    
    L, U = lu
    n = len(b)
    y = b.copy()
    for j in range(n):
        yj = y[j]
        if abs(yj) > EPS:
            start = L.indptr[j]
            end = L.indptr[j + 1]
            for idx in range(start, end):
                i = L.indices[idx]
                if i > j:
                    y[i] -= L.data[idx] * yj
    x = y.copy()
    for j in range(n - 1, -1, -1):
        start = U.indptr[j]
        end = U.indptr[j + 1]
        
        diag = 0.0
        diag_found = False
        for idx in range(start, end):
            i = U.indices[idx]
            if i == j:
                diag = U.data[idx]
                diag_found = True
                break
        
        if not diag_found or abs(diag) < EPS:
            return None
        
        x[j] /= diag
        xj = x[j]
        
        if abs(xj) > EPS:
            for idx in range(start, end):
                i = U.indices[idx]
                if i < j:
                    x[i] -= U.data[idx] * xj
    
    return x


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя LU-разложением
    det(A) = det(L) * det(U) = 1 * ∏U_ii
    """
    lu = lu_decomposition(A)
    if lu is None:
        return 0.0
    
    _, U = lu
    n = U.shape[0]
    det = 1.0
    
    for j in range(n):
        start = U.indptr[j]
        end = U.indptr[j + 1]
        
        diag_found = False
        for idx in range(start, end):
            i = U.indices[idx]
            if i == j:
                det *= U.data[idx]
                diag_found = True
                break
        
        if not diag_found:
            return 0.0
    return det