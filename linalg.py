from CSC import CSCMatrix
from CSR import CSRMatrix
from typing import Tuple, Optional, List
Vector = List[float]


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix, List[int]]]:
    """
    LU-разложение для CSC матрицы без выбора главного элемента.
    Возвращает (L, U, P) - нижнюю и верхнюю треугольные матрицы и вектор перестановок.
    Ожидается, что матрица L хранит единицы на главной диагонали.
    """
    n = A.shape[0]
    assert A.shape[0] == A.shape[1], "Matrix must be square for LU decomposition"

    dense_A = A.to_dense()

    L = [[0.0 for _ in range(n)] for _ in range(n)]
    U = [[0.0 for _ in range(n)] for _ in range(n)]

    # Без выбора главного элемента, P - единичная перестановка
    P = list(range(n))

    for i in range(n):
        # Проверка на вырожденность
        if abs(dense_A[i][i]) < 1e-12:
            # Попробуем найти ненулевой элемент ниже для перестановки
            found = False
            for k in range(i + 1, n):
                if abs(dense_A[k][i]) > 1e-12:
                    # Меняем строки
                    dense_A[i], dense_A[k] = dense_A[k], dense_A[i]
                    P[i], P[k] = P[k], P[i]
                    # Также меняем уже вычисленные строки в L
                    for j in range(i):
                        L[i][j], L[k][j] = L[k][j], L[i][j]
                    found = True
                    break
            if not found:
                return None

        # Вычисляем U[i][j]
        for j in range(i, n):
            sum_u = 0.0
            for k in range(i):
                sum_u += L[i][k] * U[k][j]
            U[i][j] = dense_A[i][j] - sum_u

        # Вычисляем L[j][i]
        for j in range(i + 1, n):
            sum_l = 0.0
            for k in range(i):
                sum_l += L[j][k] * U[k][i]
            L[j][i] = (dense_A[j][i] - sum_l) / U[i][i]

        L[i][i] = 1.0

    # Преобразуем L и U в разреженный формат CSC
    L_data, L_rows, L_cols = [], [], []
    U_data, U_rows, U_cols = [], [], []

    for i in range(n):
        for j in range(n):
            if j <= i and abs(L[i][j]) >= 1e-15:
                L_data.append(L[i][j])
                L_rows.append(i)
                L_cols.append(j)

            if j >= i and abs(U[i][j]) >= 1e-15:
                U_data.append(U[i][j])
                U_rows.append(i)
                U_cols.append(j)

    from COO import COOMatrix

    L_coo = COOMatrix(L_data, L_rows, L_cols, (n, n))
    U_coo = COOMatrix(U_data, U_rows, U_cols, (n, n))

    return L_coo._to_csc(), U_coo._to_csc(), P


def solve_lower_triangular(L: CSCMatrix, b: Vector) -> Vector:
    """Решение Ly = b (L - нижняя треугольная с единицами на диагонали)."""
    n = L.shape[0]
    y = [0.0] * n

    for i in range(n):
        sum_val = 0.0
        row_start = L.indptr[i]
        row_end = L.indptr[i + 1]

        for idx in range(row_start, row_end):
            j = L.indices[idx]
            if j < i:
                sum_val += L.data[idx] * y[j]

        y[i] = b[i] - sum_val

    return y


def solve_upper_triangular(U: CSCMatrix, y: Vector) -> Vector:
    """Решение Ux = y (U - верхняя треугольная)."""
    n = U.shape[0]
    x = [0.0] * n

    for i in range(n - 1, -1, -1):
        sum_val = 0.0
        row_start = U.indptr[i]
        row_end = U.indptr[i + 1]

        for idx in range(row_start, row_end):
            j = U.indices[idx]
            if j > i:
                sum_val += U.data[idx] * x[j]

        diag_val = 0.0
        for idx in range(row_start, row_end):
            if U.indices[idx] == i:
                diag_val = U.data[idx]
                break

        if abs(diag_val) < 1e-15:
            raise ValueError("Matrix U is singular")

        x[i] = (y[i] - sum_val) / diag_val

    return x


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None

    L, U, P = lu_result

    # Применяем перестановку к вектору b: b' = Pb
    b_permuted = [b[p] for p in P]

    y = solve_lower_triangular(L, b_permuted)
    x = solve_upper_triangular(U, y)

    return x


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = (-1)^s * det(L) * det(U) = (-1)^s * prod(U[i][i])
    где s - количество перестановок строк.
    """
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None

    L, U, P = lu_result
    n = A.shape[0]

    # Подсчет количества перестановок
    swaps = 0
    for i in range(n):
        if P[i] != i:
            # Находим, куда переставлен элемент i
            for j in range(i + 1, n):
                if P[j] == i:
                    swaps += 1
                    break

    # Вычисляем определитель U
    det = 1.0
    dense_U = U.to_dense()
    for i in range(n):
        det *= dense_U[i][i]

    # Учет знака перестановок
    if swaps % 2 == 1:
        det = -det

    return det
