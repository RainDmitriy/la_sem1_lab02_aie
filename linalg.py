from CSC import CSCMatrix
from CSR import CSRMatrix
from .types import Vector
from typing import Tuple, Optional

def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    Ожидается, что матрица L хранит единицы на главной диагонали.
    """
    n, m = A.shape
    if n != m:
        return None

    L_data, L_indices, L_indptr = [], [], [0]
    U_data, U_indices, U_indptr = [], [], [0]

    A_cols = []
    for j in range(n):
        col = {}
        for idx in range(A.indptr[j], A.indptr[j + 1]):
            col[A.indices[idx]] = A.data[idx]
        A_cols.append(col)

    U_cols = [{} for _ in range(n)]
    L_cols = [{} for _ in range(n)]

    for k in range(n):
        # U[k, k:]
        uk = dict(A_cols[k])

        for j in range(k):
            if k in L_cols[j]:
                ljk = L_cols[j][k]
                for i, val in U_cols[j].items():
                    uk[i] = uk.get(i, 0.0) - ljk * val

        if k not in uk or uk[k] == 0:
            return None  # ведущий элем

        pivot = uk[k]

        for i, val in uk.items():
            if i >= k:
                U_cols[k][i] = val

        # L[k+1:, k]
        for i in range(k + 1, n):
            if i in uk:
                L_cols[k][i] = uk[i] / pivot

        L_cols[k][k] = 1.0

    for j in range(n):
        col = L_cols[j]
        for i in sorted(col):
            L_indices.append(i)
            L_data.append(col[i])
        L_indptr.append(len(L_data))

    for j in range(n):
        col = U_cols[j]
        for i in sorted(col):
            U_indices.append(i)
            U_data.append(col[i])
        U_indptr.append(len(U_data))

    L = CSCMatrix(L_data, L_indices, L_indptr, (n, n))
    U = CSCMatrix(U_data, U_indices, U_indptr, (n, n))
    return L, U

def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    lu = lu_decomposition(A)
    if lu is None:
        return None

    L, U = lu
    n = len(b)

    y = [0.0] * n

    for i in range(n):
        s = b[i]
        for j in range(i):
            start = L.indptr[j]
            end = L.indptr[j + 1]
            for idx in range(start, end):
                if L.indices[idx] == i:
                    s -= L.data[idx] * y[j]
                    break
        y[i] = s  # L[i,i] = 1

    x = [0.0] * n

    for i in reversed(range(n)):
        s = y[i]
        diag = None

        start = U.indptr[i]
        end = U.indptr[i + 1]

        for idx in range(start, end):
            row = U.indices[idx]
            val = U.data[idx]
            if row == i:
                diag = val
            elif row > i:
                s -= val * x[row]

        if diag is None or diag == 0:
            return None

        x[i] = s / diag

    return x

def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    lu = lu_decomposition(A)
    if lu is None:
        return None

    _, U = lu
    n, _ = U.shape

    det = 1.0

    for j in range(n):
        diag = None
        start = U.indptr[j]
        end = U.indptr[j + 1]

        for idx in range(start, end):
            if U.indices[idx] == j:
                diag = U.data[idx]
                break

        if diag is None:
            return 0.0

        det *= diag

    return det
