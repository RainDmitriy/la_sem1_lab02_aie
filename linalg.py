from CSC import CSCMatrix
from type import Vector
from typing import Tuple, Optional, Dict, List


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    '''
    LU-разложение для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    Ожидается, что матрица L хранит единицы на главной диагонали.
    '''
    n = A.shape[0]
    if n != A.shape[1]:
        return None

    rows = [{} for _ in range(n)]
    for j in range(n):
        start, end = A.indptr[j], A.indptr[j + 1]
        for idx in range(start, end):
            i = A.indices[idx]
            rows[i][j] = A.data[idx]

    pivot = list(range(n))

    for k in range(n):
        max_val = 0.0
        max_row = k
        for i in range(k, n):
            val = abs(rows[i].get(k, 0.0))
            if val > max_val:
                max_val = val
                max_row = i

        if max_val < 1e-12:
            return None

        if max_row != k:
            rows[k], rows[max_row] = rows[max_row], rows[k]
            pivot[k], pivot[max_row] = pivot[max_row], pivot[k]

        pivot_val = rows[k][k]
        for i in range(k + 1, n):
            if k in rows[i]:
                factor = rows[i][k] / pivot_val
                rows[i][k] = factor

                for j in range(k + 1, n):
                    if k in rows[i] and j in rows[k]:
                        rows[i][j] = rows[i].get(j, 0.0) - factor * rows[k][j]

    L_data, L_row, L_col = [], [], []
    U_data, U_row, U_col = [], [], []

    for i in range(n):
        L_data.append(1.0)
        L_row.append(i)
        L_col.append(i)

        for j, val in rows[i].items():
            if j < i:
                L_data.append(val)
                L_row.append(i)
                L_col.append(j)
            elif j >= i:
                U_data.append(val)
                U_row.append(i)
                U_col.append(j)

    from COO import COOMatrix
    L_coo = COOMatrix(L_data, L_row, L_col, (n, n))
    U_coo = COOMatrix(U_data, U_row, U_col, (n, n))

    return (L_coo._to_csc(), U_coo._to_csc())


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    '''
    Решение СЛАУ Ax = b через LU-разложение.
    '''
    result = lu_decomposition(A)
    if result is None:
        return None

    L, U = result
    n = len(b)

    y = [0.0] * n
    dense_L = L.to_dense()

    for i in range(n):
        s = b[i]
        for j in range(i):
            s -= dense_L[i][j] * y[j]
        y[i] = s

    x = [0.0] * n
    dense_U = U.to_dense()

    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= dense_U[i][j] * x[j]

        if abs(dense_U[i][i]) < 1e-12:
            return None

        x[i] = s / dense_U[i][i]

    return x


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    '''
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    '''
    result = lu_decomposition(A)
    if result is None:
        return None

    _, U = result
    dense_U = U.to_dense()
    det = 1.0
    n = A.shape[0]

    for i in range(n):
        det *= dense_U[i][i]

    return det