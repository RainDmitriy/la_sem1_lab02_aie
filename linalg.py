from CSC import CSCMatrix
from CSR import CSRMatrix
from types import Vector
from typing import Tuple, Optional

EPSILON = 1e-10

def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    Ожидается, что матрица L хранит единицы на главной диагонали.
    """
    n = A.shape[0]
    dense = A.to_dense()

    # пустые L и U
    L = [[0] * n for _ in range(n)]
    U = [[0] * n for _ in range(n)]

    for k in range(n):
        # сначала устанавливаем единицу на диагонали
        L[k][k] = 1.0

        # считаем U_{k,j}
        for j in range(k, n):
            s = 0.0
            # складываем до k-1
            for p in range(k):
                s += L[k][p] * U[p][j]
            U[k][j] = dense[k][j] - s

        # если диагональный элемент U слишком маленький,
        # считаем матрицу вырожденной
        if abs(U[k][k]) < EPSILON:
            return None

        for i in range(k + 1, n):
            s = 0
            for p in range(k):
                s += L[i][p] * U[p][k]
            L[i][k] = (dense[i][k] - s) / U[k][k]
    
    return CSCMatrix.from_dense(L), CSCMatrix.from_dense(U)

def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    # получаем разложение
    lu_result = lu_decomposition(A)

    # если матрица вырождена
    if lu_result is None:
        return None
    
    L, U = lu_result
    n = A.shape[0]
    
    # преобразуем L и U в плотные для удобства
    L_dense = L.to_dense()
    U_dense = U.to_dense()
    
    # решаем Ly = b
    y = [0] * n
    for i in range(n):
        s = 0
        # складываем
        for j in range(i):
            s += L_dense[i][j] * y[j]

        y[i] = b[i] - s
    
    # решаем Ux = y
    x = [0] * n
    for i in range(n - 1, -1, -1):
        s = 0
        # складываем
        for j in range(i + 1, n):
            s += U_dense[i][j] * x[j]
        
        # если диагональный элемент равен нулю
        if abs(U_dense[i][i]) < EPSILON:
            return None  # матрица вырождена
        
        x[i] = (y[i] - s) / U_dense[i][i]
    
    return x

def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    # получаем разложение
    lu_result = lu_decomposition(A)

    # если матрица вырождена
    if lu_result is None:
        return None
    
    L, U = lu_result
    n = A.shape[0]
    
    # преобразуем U в плотную матрицу
    U_dense = U.to_dense()
    
    det_L = 1.0
    
    # произведение диагональных элементов будет детерминантом
    det_U = 1
    for i in range(n):
        det_U *= U_dense[i][i]
    
    return det_L * det_U

