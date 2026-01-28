from CSC import CSCMatrix
from CSR import CSRMatrix
from type import Vector
from typing import Tuple, Optional

EPSILON = 1e-10


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    Ожидается, что матрица L хранит единицы на главной диагонали.
    """
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("LU‑разложение определено только для квадратных матриц")
    L_cols: list[dict[int, float]] = [{} for _ in range(n)]
    U_rows: list[dict[int, float]] = [{} for _ in range(n)]
    rows_A: list[dict[int, float]] = [{} for _ in range(n)]
    for col in range(n):
        for idx in range(A.indptr[col], A.indptr[col + 1]):
            row = A.indices[idx]
            rows_A[row][col] = float(A.data[idx])
    active_rows: list[dict[int, float]] = [{} for _ in range(n)]
    for k in range(n):
        row_k: dict[int, float] = {}
        for j, val in rows_A[k].items():
            if j >= k:
                row_k[j] = float(val)
        for j, val in active_rows[k].items():
            if j >= k:
                current = row_k.get(j, 0.0)
                row_k[j] = current + float(val)
        u_kk = row_k.get(k, 0.0)
        if abs(u_kk) < EPSILON:
            return None
        U_rows[k] = {}
        for j, val in row_k.items():
            if j >= k and abs(val) > EPSILON:
                U_rows[k][j] = val
        U_rows[k][k] = u_kk
        L_cols[k][k] = 1.0
        for i in range(k + 1, n):
            elem = 0.0
            if k in rows_A[i]:
                elem += float(rows_A[i][k])
            if k in active_rows[i]:
                elem += float(active_rows[i][k])
            if abs(elem) > EPSILON:
                L_ik = elem / u_kk
                L_cols[k][i] = L_ik
                for j, U_kj in U_rows[k].items():
                    if j > k:
                        delta = -L_ik * U_kj
                        if abs(delta) > EPSILON:
                            current = active_rows[i].get(j, 0.0)
                            active_rows[i][j] = current + delta
    L_data: list[float] = []
    L_indices: list[int] = []
    L_indptr: list[int] = [0]
    for col in range(n):
        rows = sorted(L_cols[col].keys())
        for i in rows:
            L_data.append(L_cols[col][i])
            L_indices.append(i)
        L_indptr.append(len(L_data))
    U_cols: list[dict[int, float]] = [{} for _ in range(n)]
    for i in range(n):
        for j, val in U_rows[i].items():
            U_cols[j][i] = val
    U_data: list[float] = []
    U_indices: list[int] = []
    U_indptr: list[int] = [0]
    for col in range(n):
        rows = sorted(U_cols[col].keys())
        for i in rows:
            U_data.append(U_cols[col][i])
            U_indices.append(i)
        U_indptr.append(len(U_data))
    return CSCMatrix(L_data, L_indices, L_indptr, (n, n)), CSCMatrix(U_data, U_indices, U_indptr, (n, n))


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    lu = lu_decomposition(A)
    if lu is None:
        return None
    L, U = lu
    n = A.shape[0]
    if len(b) != n:
        raise ValueError("Размерность вектора b должна соответствовать матрице")
    L_dense = L.to_dense()
    U_dense = U.to_dense()
    y: Vector = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L_dense[i][j] * y[j]
        y[i] = b[i] - s
    x: Vector = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += U_dense[i][j] * x[j]
        if abs(U_dense[i][i]) < EPSILON:
            return None
        x[i] = (y[i] - s) / U_dense[i][i]
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
    n = A.shape[0]
    U_dense = U.to_dense()
    det = 1.0
    for i in range(n):
        det *= U_dense[i][i]
    return det