from CSC import CSCMatrix
from CSR import CSRMatrix
from matrix_types import Vector
from typing import Tuple

def lu_decomposition(A: CSCMatrix) -> Tuple[CSCMatrix, CSCMatrix]:
    """
    LU-разложение для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    Ожидается, что матрица L хранит единицы на главной диагонали.
    """
    n = A.shape[0]
    L_data, L_indices, L_indptr = [], [], [0]
    U_data, U_indices, U_indptr = [], [], [0]

    col_vals = [0.0] * n
    col_marks = [False] * n

    for j in range(n):
        # Копируем j-й столбец
        for idx in range(A.indptr[j], A.indptr[j+1]):
            row = A.indices[idx]
            val = A.data[idx]
            col_vals[row] = val
            col_marks[row] = True

        # Вычитаем L*U
        for idx in range(L_indptr[j], len(L_data)):
            i = L_indices[idx]
            if i >= j:
                break
            factor = L_data[idx]
            for u_idx in range(U_indptr[i], U_indptr[i+1]):
                u_col = U_indices[u_idx]
                if col_marks[u_col]:
                    col_vals[u_col] -= factor * U_data[u_idx]

        diag = col_vals[j]  # предполагаем, что диагональ ненулевая

        # U: верхняя треугольная часть
        for i in range(j, n):
            if col_marks[i] and col_vals[i] != 0:
                U_data.append(col_vals[i])
                U_indices.append(i)
        U_indptr.append(len(U_data))

        # L: нижняя треугольная часть (единицы на диагонали логически)
        for i in range(j+1, n):
            if col_marks[i] and col_vals[i] != 0:
                L_data.append(col_vals[i] / diag)
                L_indices.append(i)
        L_indptr.append(len(L_data))

        # Очистка scratch
        for i in range(n):
            if col_marks[i]:
                col_vals[i] = 0.0
                col_marks[i] = False

    L = CSCMatrix(n, n, L_data, L_indices, L_indptr)
    U = CSCMatrix(n, n, U_data, U_indices, U_indptr)
    return L, U


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Vector:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    L, U = lu_decomposition(A)
    n = len(b)

    # Прямой ход Ly=b
    y = b[:]
    for j in range(n):
        for idx in range(L.indptr[j], L.indptr[j+1]):
            i = L.indices[idx]
            y[i] -= L.data[idx] * y[j]

    # Обратный ход Ux=y
    x = y[:]
    for j in reversed(range(n)):
        diag = None
        for idx in range(U.indptr[j], U.indptr[j+1]):
            if U.indices[idx] == j:
                diag = U.data[idx]
                break
        x[j] /= diag
        for idx in range(U.indptr[j], U.indptr[j+1]):
            i = U.indices[idx]
            if i < j:
                x[i] -= U.data[idx] * x[j]

    return x


def find_det_with_lu(A: CSCMatrix) -> float:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    _, U = lu_decomposition(A)
    det = 1.0
    n = U.shape[0]
    for j in range(n):
        for idx in range(U.indptr[j], U.indptr[j+1]):
            if U.indices[idx] == j:
                det *= U.data[idx]
                break
    return det
