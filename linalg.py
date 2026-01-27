from CSC import CSCMatrix
from type import Vector
from typing import Tuple, Optional

def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    n = A.shape[0]
    if n != A.shape[1]: return None
    
    A_dense = A.to_dense()
    L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    U = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(i, n):
            U[i][j] = A_dense[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
        for j in range(i + 1, n):
            if abs(U[i][i]) < 1e-15: return None # Вырожденная матрица
            L[j][i] = (A_dense[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]

    return CSCMatrix.from_dense(L), CSCMatrix.from_dense(U)

def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    res = lu_decomposition(A)
    if not res: return None
    L_obj, U_obj = res
    L, U = L_obj.to_dense(), U_obj.to_dense()
    n = len(b)
    
    # Решаем Ly = b (прямой ход)
    y = [0.0] * n
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))
        
    # Решаем Ux = y (обратный ход)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(U[i][i]) < 1e-15: return None
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
        
    return x

def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    res = lu_decomposition(A)
    if not res: return 0.0
    U = res[1].to_dense()
    det = 1.0
    for i in range(len(U)):
        det *= U[i][i]
    return det
