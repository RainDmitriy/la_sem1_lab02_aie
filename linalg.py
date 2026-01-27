from typing import List, Tuple, Optional
from CSC import CSCMatrix
from CSR import CSRMatrix

Vector = List[float]


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    Ожидается, что матрица L хранит единицы на главной диагонали.
    """
    if A.rows != A.cols:
        return None
    
    n = A.rows
    dense = A.to_dense()
    
    # Создаём копии матриц L и U
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        # Верхняя треугольная матрица U
        for j in range(i, n):
            s = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = dense[i][j] - s
        
        # Нижняя треугольная матрица L (с единицами на диагонали)
        L[i][i] = 1.0
        for j in range(i + 1, n):
            s = sum(L[j][k] * U[k][i] for k in range(i))
            if U[i][i] == 0:
                return None  # Матрица вырождена
            L[j][i] = (dense[j][i] - s) / U[i][i]
    
    # Преобразуем L и U в CSC
    L_csc = CSCMatrix.from_dense(L)
    U_csc = CSCMatrix.from_dense(U)
    
    return L_csc, U_csc


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None
    
    L, U = lu_result
    n = len(b)
    
    # Прямой ход: Ly = b
    y = [0.0] * n
    for i in range(n):
        s = 0.0
        # Получаем i-ю строку L
        for j in range(i):
            # Ищем элемент L[i][j]
            col_start = L.indptr[j]
            col_end = L.indptr[j + 1]
            for k in range(col_start, col_end):
                if L.indices[k] == i:
                    s += L.data[k] * y[j]
                    break
        y[i] = b[i] - s
    
    # Обратный ход: Ux = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = 0.0
        # Получаем i-ю строку U
        for j in range(i + 1, n):
            # Ищем элемент U[i][j]
            col_start = U.indptr[j]
            col_end = U.indptr[j + 1]
            for k in range(col_start, col_end):
                if U.indices[k] == i:
                    s += U.data[k] * x[j]
                    break
        
        # Ищем диагональный элемент U[i][i]
        diag = 0.0
        col_start = U.indptr[i]
        col_end = U.indptr[i + 1]
        for k in range(col_start, col_end):
            if U.indices[k] == i:
                diag = U.data[k]
                break
        
        if diag == 0:
            return None
        
        x[i] = (y[i] - s) / diag
    
    return x


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None
    
    _, U = lu_result
    det = 1.0
    
    # Определитель U - произведение диагональных элементов
    for i in range(A.rows):
        # Ищем диагональный элемент U[i][i]
        found = False
        col_start = U.indptr[i]
        col_end = U.indptr[i + 1]
        for k in range(col_start, col_end):
            if U.indices[k] == i:
                det *= U.data[k]
                found = True
                break
        if not found:
            det *= 0.0
    
    return det
