from CSC import CSCMatrix
from types import Vector, DenseMatrix
from typing import Tuple, Optional

def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение методом Дулиттла.
    """
    n_rows, n_cols = A.shape
    if n_rows != n_cols:
        return None 
    
    matrix = A.to_dense()
    
    L = [[0.0] * n_rows for _ in range(n_rows)]
    U = [[0.0] * n_rows for _ in range(n_rows)]

    

    for i in range(n_rows):
        # 1. Считаем U (верхнюю треугольную)
        for k in range(i, n_rows):
            sum_upper = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = matrix[i][k] - sum_upper

        # 2. Считаем L (нижнюю треугольную)
        for k in range(i, n_rows):
            if i == k:
                L[i][i] = 1.0
            else:
                sum_lower = sum(L[k][j] * U[j][i] for j in range(i))
                
                if U[i][i] == 0:
                    return None 
                    
                L[k][i] = (matrix[k][i] - sum_lower) / U[i][i]

    return CSCMatrix.from_dense(L), CSCMatrix.from_dense(U)

def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    # Проверка размеров
    if A.shape[0] != A.shape[1] or len(b) != A.shape[0]:
        return None

    lu_res = lu_decomposition(A)
    if lu_res is None:
        return None
        
    L_csc, U_csc = lu_res
    L = L_csc.to_dense()
    U = U_csc.to_dense()
    n = len(b)

    # 1. Прямой ход: Ly = b
    y = [0.0] * n
    for i in range(n):
        sum_val = sum(L[i][j] * y[j] for j in range(i))
        y[i] = b[i] - sum_val

    # 2. Обратный ход: Ux = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_val = sum(U[i][j] * x[j] for j in range(i + 1, n))
        
        if U[i][i] == 0:
            return None
            
        x[i] = (y[i] - sum_val) / U[i][i]

    return x

def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя.
    """
    lu_res = lu_decomposition(A)
    if lu_res is None:
        return None
        
    _, U_csc = lu_res
    U = U_csc.to_dense()
    
    determinant = 1.0
    for i in range(len(U)):
        determinant *= U[i][i]
        
    return determinant