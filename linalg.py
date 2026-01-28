from CSC import CSCMatrix
from CSR import CSRMatrix
from COO import COOMatrix
from type import Vector
from typing import Tuple, Optional, List, Dict

EPSILON = 1e-12


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix, List[int]]]:
    n = A.shape[0]
    
    if n != A.shape[1]:
        raise ValueError("LU-разложение требует квадратную матрицу")
    
    # Получаем элементы матрицы
    matrix = [[0.0] * n for _ in range(n)]
    for j in range(n):
        for idx in range(A.indptr[j], A.indptr[j + 1]):
            i = A.indices[idx]
            matrix[i][j] = A.data[idx]
    
    # Рабочие копии для L и U
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    P = list(range(n))
    
    # Для каждой строки
    for i in range(n):
        L[i][i] = 1.0
    
    # Копируем A в U
    for i in range(n):
        for j in range(n):
            U[i][j] = matrix[i][j]
    
    # Прямой ход метода Гаусса
    for k in range(n):
        # Выбор ведущего элемента
        max_row = k
        max_val = abs(U[k][k])
        
        for i in range(k + 1, n):
            if abs(U[i][k]) > max_val:
                max_val = abs(U[i][k])
                max_row = i
        
        if max_val < EPSILON:
            return None
        
        # Перестановка строк
        if max_row != k:
            # Меняем строки в U
            U[k], U[max_row] = U[max_row], U[k]
            # Меняем строки в L (только уже вычисленные столбцы)
            for j in range(k):
                L[k][j], L[max_row][j] = L[max_row][j], L[k][j]
            # Меняем перестановку
            P[k], P[max_row] = P[max_row], P[k]
        
        # Обновление L и U
        for i in range(k + 1, n):
            if abs(U[k][k]) < EPSILON:
                return None
            
            L[i][k] = U[i][k] / U[k][k]
            for j in range(k, n):
                U[i][j] -= L[i][k] * U[k][j]
    
    # Очистка нулей
    for i in range(n):
        for j in range(n):
            if abs(L[i][j]) < EPSILON:
                L[i][j] = 0.0
            if abs(U[i][j]) < EPSILON:
                U[i][j] = 0.0
    
    # Конвертация в CSC
    L_csc = CSCMatrix.from_dense(L)
    U_csc = CSCMatrix.from_dense(U)
    
    return L_csc, U_csc, P


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    lu_result = lu_decomposition(A)
    
    if lu_result is None:
        return None
    
    L, U, P = lu_result
    n = A.shape[0]
    
    # Применяем перестановку
    b_perm = [b[P[i]] for i in range(n)]
    
    # Решение Ly = b_perm
    y = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L.get(i, j) * y[j]
        y[i] = b_perm[i] - s
    
    # Решение Ux = y
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
    
    # Знак перестановки
    swaps = 0
    for i in range(n):
        if P[i] != i:
            for j in range(i + 1, n):
                if P[j] == i:
                    P[i], P[j] = P[j], P[i]
                    swaps += 1
                    break
    
    sign = 1 if swaps % 2 == 0 else -1
    
    # Определитель
    det = sign
    for i in range(n):
        diag = U.get(i, i)
        if abs(diag) < EPSILON:
            return 0.0
        det *= diag
    
    return det