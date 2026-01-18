from CSC import CSCMatrix
from CSR import CSRMatrix
from matrix_types import Vector
from typing import Tuple, Optional

def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    Ожидается, что матрица L хранит единицы на главной диагонали.
    """
    n_rows, n_cols = A.shape
    if n_rows != n_cols:
        return None  # LU только для квадратных матриц

    # I. Преобразование в плотную матрицу
    dense_A = A.to_dense()

    n = n_rows
    L = [[0.0 for _ in range(n)] for _ in range(n)]
    U = [[0.0 for _ in range(n)] for _ in range(n)]

    # II. Наивное LU-разложение
    for i in range(n):
        # Вычисляем элементы строки i матрицы U
        for j in range(i, n):
            sum_val = 0.0
            for k in range(i):
                sum_val += L[i][k] * U[k][j]
            U[i][j] = dense_A[i][j] - sum_val

        # Вычисляем элементы столбца i матрицы L
        for j in range(i, n):
            if i == j:
                L[i][i] = 1.0  # диагональ L — единицы
            else:
                sum_val = 0.0
                for k in range(i):
                    sum_val += L[j][k] * U[k][i]
                if abs(U[i][i]) < 1e-12:
                    return None  # деление на ноль — разложение невозможно
                L[j][i] = (dense_A[j][i] - sum_val) / U[i][i]

    # III. Преобразование L и U в CSC
    # Для L: явное хранение единиц на диагонали
    L_csc = CSCMatrix.from_dense(L)
    U_csc = CSCMatrix.from_dense(U)

    return L_csc, U_csc


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    lu = lu_decomposition(A)
    if lu is None:
        return None

    L, U = lu
    n = len(b)

    # Прямой ход: Ly = b
    y = [0.0] * n
    L_dense = L.to_dense()
    for i in range(n):
        sum_val = 0.0
        for j in range(i):
            sum_val += L_dense[i][j] * y[j]
        y[i] = b[i] - sum_val  # L[i][i] = 1

    # Обратный ход: Ux = y
    x = [0.0] * n
    U_dense = U.to_dense()
    for i in range(n - 1, -1, -1):
        sum_val = 0.0
        for j in range(i + 1, n):
            sum_val += U_dense[i][j] * x[j]
        if abs(U_dense[i][i]) < 1e-12:
            return None
        x[i] = (y[i] - sum_val) / U_dense[i][i]

    return x

def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U) = 1 * prod(diag(U)) = prod(U[i][i])
    """
    lu = lu_decomposition(A)
    if lu is None:
        return None

    L, U = lu
    U_dense = U.to_dense()
    det = 1.0
    n = U.shape[0]
    for i in range(n):
        det *= U_dense[i][i]
        if abs(det) < 1e-100:  # избегаем underflow
            break
    return det

