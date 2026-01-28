from CSC import CSCMatrix
from CSR import CSRMatrix
from .types import Vector
from typing import Tuple, Optional


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """LU-разложение."""
    size, _ = A.shape
    if size != A.shape[1]:
        return None

    raw = A.to_dense()
    low = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
    up = [[0.0 for _ in range(size)] for _ in range(size)]

    for i in range(size):
        # Верхняя
        curr_row = i
        while curr_row < size:
            sum_val = 0.0
            for k in range(i):
                sum_val += low[i][k] * up[k][curr_row]
            up[i][curr_row] = raw[i][curr_row] - sum_val
            curr_row += 1

        # Нижняя
        curr_col = i + 1
        while curr_col < size:
            if abs(up[i][i]) < 1e-16:
                return None
            sum_val = 0.0
            for k in range(i):
                sum_val += low[curr_col][k] * up[k][i]
            low[curr_col][i] = (raw[curr_col][i] - sum_val) / up[i][i]
            curr_col += 1

    return CSCMatrix.from_dense(low), CSCMatrix.from_dense(up)


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """Решение через LU."""
    decomp = lu_decomposition(A)
    if not decomp:
        return None
    l_obj, u_obj = decomp

    n = A.shape[0]
    if len(b) != n:
        return None

    l_mtx, u_mtx = l_obj.to_dense(), u_obj.to_dense()

    # Прямой
    y = [0.0] * n
    for i in range(n):
        tmp = sum(l_mtx[i][j] * y[j] for j in range(i))
        y[i] = b[i] - tmp

    # Обратный
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(u_mtx[i][i]) < 1e-16:
            return None
        tmp = sum(u_mtx[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - tmp) / u_mtx[i][i]

    return x


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """Определитель."""
    res = lu_decomposition(A)
    if res is None:
        return None
    _, u_part = res
    u_data = u_part.to_dense()

    total = 1.0
    for idx in range(len(u_data)):
        total *= u_data[idx][idx]
    return total