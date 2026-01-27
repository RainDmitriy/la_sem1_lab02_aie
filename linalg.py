from typing import Tuple, Optional, List
from CSC import CSCMatrix


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    """
    if A.rows != A.cols:
        return None
    
    n = A.rows
    dense = A.to_dense()
    
    # Создаем копии для L и U
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    
    # Копируем A в U для начала
    for i in range(n):
        for j in range(n):
            U[i][j] = dense[i][j]
    
    # Выполняем LU-разложение
    for k in range(n):
        if abs(U[k][k]) < 1e-12:
            return None
        
        L[k][k] = 1.0
        
        for i in range(k + 1, n):
            L[i][k] = U[i][k] / U[k][k]
            for j in range(k, n):
                U[i][j] -= L[i][k] * U[k][j]
    
    # Преобразуем в CSC
    L_csc = CSCMatrix.from_dense(L)
    U_csc = CSCMatrix.from_dense(U)
    
    return L_csc, U_csc


def solve_SLAE_lu(A: CSCMatrix, b: List[float]) -> Optional[List[float]]:
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
    dense_L = L.to_dense()
    
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += dense_L[i][j] * y[j]
        y[i] = b[i] - s
    
    # Обратный ход: Ux = y
    x = [0.0] * n
    dense_U = U.to_dense()
    
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += dense_U[i][j] * x[j]
        
        if abs(dense_U[i][i]) < 1e-12:
            return None
        
        x[i] = (y[i] - s) / dense_U[i][i]
    
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
    dense_U = U.to_dense()
    
    for i in range(A.rows):
        det *= dense_U[i][i]
    
    return det
