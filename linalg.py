from CSC import CSCMatrix
from CSR import CSRMatrix
from type import Vector
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
    dense = A.to_dense()
    L = [[0.0]*n for _ in range(n)]
    U = [[0.0]*n for _ in range(n)]
    for i in range(n):
        L[i][i] = 1.0
    for k in range(n):
        for j in range(k, n):
            s = sum(L[k][p] * U[p][j] for p in range(k))
            U[k][j] = dense[k][j] - s
        if U[k][k] == 0:
            return None
        for i in range(k+1, n):
            s = sum(L[i][p] * U[p][k] for p in range(k))
            L[i][k] = (dense[i][k] - s) / U[k][k]
    return CSCMatrix.from_dense(L), CSCMatrix.from_dense(U)

def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    n, m = A.shape
    if n != m:
        return None
    A = A._to_csr()
    rows = {}
    for i in range(n):
        row_dict = {}
        for idx in range(A.indptr[i], A.indptr[i+1]):
            col = A.indices[idx]
            row_dict[col] = A.data[idx]
        rows[i] = row_dict
    b = list(b)

    for k in range(n):
        if k not in rows[k] or rows[k][k] == 0:
            return None
        pivot = rows[k][k]
        for i in range(k+1, n):
            if k in rows[i]:
                factor = rows[i][k] / pivot
                for j, val in rows[k].items():
                    rows[i][j] = rows[i].get(j, 0) - factor * val
                    if abs(rows[i][j]) < 1e-12:
                        rows[i].pop(j, None)
                b[i] -= factor * b[k]

    x = [0.0]*n
    for i in reversed(range(n)):
        if i not in rows[i] or rows[i][i] == 0:
            return None
        s = b[i]
        for j, val in rows[i].items():
            if j > i:
                s -= val * x[j]
        x[i] = s / rows[i][i]

    return x

def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    result = lu_decomposition(A)
    if result is None:
        return None
    _, U = result
    Ud = U.to_dense()
    det = 1.0
    for i in range(len(Ud)):
        det *= Ud[i][i]
    return det
