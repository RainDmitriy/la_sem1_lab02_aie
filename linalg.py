from CSC import CSCMatrix
from CSR import CSRMatrix
from types import Vector, DenseMatrix
from typing import Tuple, Optional

def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    Ожидается, что матрица L хранит единицы на главной диагонали.
    """
    dense = A.to_dense()
    n, m = A.shape

    if n != m:
        raise ValueError("LU-разложение возможно для квадратных матриц")

    L = [[0] * n for _ in range(n)]
    U = [[0] * n for _ in range(n)]

    for i in range(n):
        # U
        for j in range(i, n):
            sum_u = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = dense.data[i][j] - sum_u

        # L
        for j in range(i, n):
            if i == j:
                L[i][i] = 1
            else:
                sum_l = sum(L[j][k] * U[k][i] for k in range(i))
                if U[i][i] == 0:
                    raise ValueError("Разложение невозможно")
                L[j][i] = (dense.data[j][i] - sum_l) / U[i][i]

    L = CSCMatrix.from_dense(DenseMatrix(L))
    U = CSCMatrix.from_dense(DenseMatrix(U))
    return L, U

def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    n, m = A.shape
    if n != m:
        raise ValueError("Матрица должна быть квадратной")
    if len(b) != n:
        raise ValueError("Размер вектора не совпадает с размерностью матрицы")

    L, U = lu_decomposition(A)
    L = L.to_dense().data
    U = U.to_dense().data

    # Ly = b
    y = [0] * n
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))

    # Ux = y
    x = [0] * n
    for i in range(n, -1, -1):
        if U[i][i] == 0:
            raise ValueError("Матрица вырождена, решения нет")
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i+1, n))) / U[i][i]

    return x

def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("Матрица должна быть квадратной")
    
    _, U = lu_decomposition(A)
    U = U.to_dense().data

    det = 1
    for i in range(A.shape[0]):
        det *= U[i][i]

    return det