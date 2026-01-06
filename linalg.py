from CSC import CSCMatrix
from CSR import CSRMatrix
from types import Vector
from typing import Tuple, Optional

def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    Ожидается, что матрица L хранит единицы на главной диагонали.
    """
    n = A.shape[0]
    dense_A = A.to_dense()
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for k in range(n):
        if abs(dense_A[k][k]) < 1e-15:
            found = False
            for i in range(k + 1, n):
                if abs(dense_A[i][k]) > 1e-15:
                    dense_A[k], dense_A[i] = dense_A[i], dense_A[k]
                    if k > 0:
                        L[k][:k], L[i][:k] = L[i][:k], L[k][:k]
                    found = True
                    break
            
            if not found:
                return None

    for i in range(n):
        for k in range(i, n):
            sum_val = 0.0
            for j in range(i):
                sum_val += L[i][j] * U[j][k]
            U[i][k] = dense_A[i][k] - sum_val

        for k in range(i, n):
            if i == k:
                L[i][i] = 1.0
            else:
                sum_val = 0.0
                for j in range(i):
                    sum_val += L[k][j] * U[j][i]
                
                if abs(U[i][i]) < 1e-15:
                    return None
                
                L[k][i] = (dense_A[k][i] - sum_val) / U[i][i]

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
    n = A.shape[0]
    L_dense = L.to_dense()
    U_dense = U.to_dense()

    if len(b) != n:
        raise ValueError(f"Длина вектора b ({len(b)}) не равна размеру матрицы A ({n})")

    y = [0.0] * n
    for i in range(n):
        sum_val = 0.0
        for j in range(i):
            sum_val += L_dense[i][j] * y[j]
        y[i] = b[i] - sum_val

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_val = 0.0
        for j in range(i + 1, n):
            sum_val += U_dense[i][j] * x[j]
        
        if abs(U_dense[i][i]) < 1e-15:
            return None
        
        x[i] = (y[i] - sum_val) / U_dense[i][i]
    
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
    U_dense = U.to_dense()
    det = 1.0
    n = A.shape[0]
    
    for i in range(n):
        det *= U_dense[i][i]
    
    return det
