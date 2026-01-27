from CSC import CSCMatrix
from type import Vector
from typing import Tuple, Optional

def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    n = A.shape[0]
    if n != A.shape[1]: return None
    
    mat = A.to_dense()
    L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i, n):
            # Вычисление элементов U
            U[i][j] = mat[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
        
        for j in range(i + 1, n):
            # Вычисление элементов L
            if abs(U[i][i]) < 1e-15: return None
            L[j][i] = (mat[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]
            
    return CSCMatrix.from_dense(L), CSCMatrix.from_dense(U)

def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    decomp = lu_decomposition(A)
    if not decomp: return None
    
    l_dense, u_dense = decomp[0].to_dense(), decomp[1].to_dense()
    size = len(b)
    
    # Прямая подстановка для Ly = b
    y = [0.0] * size
    for i in range(size):
        sum_val = sum(l_dense[i][j] * y[j] for j in range(i))
        y[i] = b[i] - sum_val
        
    # Обратная подстановка для Ux = y
    x = [0.0] * size
    for i in range(size - 1, -1, -1):
        if abs(u_dense[i][i]) < 1e-15: return None
        sum_val = sum(u_dense[i][j] * x[j] for j in range(i + 1, size))
        x[i] = (y[i] - sum_val) / u_dense[i][i]
        
    return x

def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    res = lu_decomposition(A)
    if not res: return 0.0
    u_mat = res[1].to_dense()
    # Определитель — произведение диагональных элементов U
    determinant = 1.0
    for i in range(len(u_mat)):
        determinant *= u_mat[i][i]
    return determinant
