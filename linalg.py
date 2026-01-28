from CSC import CSCMatrix
from COO import COOMatrix
from CSR import CSRMatrix
from type import Vector
from typing import Tuple, Optional

TOLERANCE = 1e-12


def value(csc: CSCMatrix, row: int, col: int) -> float:
    """получение элемента a[row, col] из CSC матрицы."""
    start = csc.indptr[col]
    end = csc.indptr[col + 1]
    for idx in range(start, end):
        if csc.indices[idx] == row:
            return csc.data[idx]
    return 0.0

def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    Ожидается, что матрица L хранит единицы на главной диагонали.
    """
    rows, cols = A.shape
    if rows != cols:
        return None
    n = rows
    L_data, L_row, L_col = [], [], []
    L_values = {}
    U_data, U_row, U_col = [], [], []
    U_values = {}

    for j in range(n):
        for i in range(j, n):
            val = value(A, j, i)
            for k in range(j):
                l_jk = L_values.get((j, k), 0.0)
                u_ki = U_values.get((k, i), 0.0)
                val -= l_jk * u_ki
            if abs(val) > TOLERANCE:
                U_values[(j, i)] = val
                U_row.append(j)
                U_data.append(val)
                U_col.append(i)
        u_jj = U_values.get((j, j), 0.0)
        if abs(u_jj) < TOLERANCE:
            return None
        L_row.append(j)
        L_col.append(j)
        L_values[(j, j)] = 1.0
        L_data.append(1.0)

        for i in range(j + 1, n):
            val = value(A, i, j)
            for k in range(j):
                u_kj = U_values.get((k, j), 0.0)
                l_ik = L_values.get((i, k), 0.0)
                val -= l_ik * u_kj
            val /= u_jj
            if abs(val) > TOLERANCE:
                L_values[(i, j)] = val
                L_data.append(val)
                L_row.append(i)
                L_col.append(j)

    L_coo = COOMatrix(L_data, L_row, L_col, (n, n))
    U_coo = COOMatrix(U_data, U_row, U_col, (n, n))
    return (L_coo._to_csc(), U_coo._to_csc())

def _solve_l(L: CSCMatrix, b: Vector) -> Vector:
    """решение L*y = b (CSC)"""
    n = len(b)
    y = [0.0] * n
    L_csr = L._to_csr()

    for i in range(n):
        total = b[i]
        start = L_csr.indptr[i]
        end = L_csr.indptr[i + 1]
        for idx in range(start, end):
            col = L_csr.indices[idx]
            if col < i:
                total -= L_csr.data[idx] * y[col]
        y[i] = total
    return y


def _solve_u(U: CSCMatrix, y: Vector) -> Vector:
    """решение U*x = y (CSC)"""
    U_csr = U._to_csr()
    n = len(y)
    x = [0.0] * n

    for i in range(n - 1, -1, -1):
        start = U_csr.indptr[i]
        end = U_csr.indptr[i + 1]
        total = y[i]
        diag_val = 0.0
        for idx in range(start, end):
            col = U_csr.indices[idx]
            if col == i:
                diag_val = U_csr.data[idx]
            elif col > i:
                total -= U_csr.data[idx] * x[col]
        if abs(diag_val) < TOLERANCE:
            return [float('nan')] * n
        x[i] = total / diag_val
    return x


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    lu = lu_decomposition(A)
    if lu is None:
        return None

    L, U = lu
    y = _solve_l(L, b)
    x = _solve_u(U, y)
    return x

def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    rows, cols = A.shape
    if rows != cols:
        return None
    n = rows
    lu = lu_decomposition(A)
    if lu is None:
        return None
    _, U = lu
    det = 1.0

    for i in range(n):
        diagonal = 0.0
        start = U.indptr[i]
        end = U.indptr[i + 1]
        for idx in range(start, end):
            if U.indices[idx] == i:
                diagonal = U.data[idx]
                break
        if abs(diagonal) < TOLERANCE:
            return 0.0
        det *= diagonal
    return det


