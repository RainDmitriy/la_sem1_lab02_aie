from CSC import CSCMatrix
from type import Vector
from typing import Tuple, Optional

def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU разложение для CSC матрицы
    """
    sr, sc = A.shape
    if sr != sc:
        return None

    dm = A.to_dense()
    e = 1e-12

    L = [[0.0 for _ in range(sr)] for _ in range(sr)]
    U = [[0.0 for _ in range(sr)] for _ in range(sr)]

    for r in range(sr):
        # считаем U
        for c in range(r, sr):
            s = 0.0
            for k in range(r):
                s += L[r][k] * U[k][c]
            U[r][c] = dm[r][c] - s

        p = U[r][r]
        if abs(p) < e:
            return None

        L[r][r] = 1.0

        # считаем L
        for i in range(r + 1, sr):
            s = 0.0
            for k in range(r):
                s += L[i][k] * U[k][r]
            L[i][r] = (dm[i][r] - s) / p

    return CSCMatrix.from_dense(L), CSCMatrix.from_dense(U)

def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение Ax=b через LU
    """
    lu_d = lu_decomposition(A)
    if lu_d is None:
        return None
    l, u = lu_d

    sr, sc = A.shape
    if sr != sc or len(b) != sr:
        return None

    ld = l.to_dense()
    ud = u.to_dense()

    y = [0.0 for _ in range(sr)]
    for r in range(sr):
        s = 0.0
        for c in range(r):
            s += ld[r][c] * y[c]
            
        y[r] = float(b[r]) - s

    x = [0.0 for _ in range(sr)]
    e = 1e-12
    for r in range(sr - 1, -1, -1):
        s = 0.0
        for c in range(r + 1, sr):
            s += ud[r][c] * x[c]
        if abs(ud[r][r]) < e:
            return None
        x[r] = (y[r] - s) / ud[r][r]

    return x

def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Вычисление определителя через LU
    """
    lu_d = lu_decomposition(A)
    if lu_d is None:
        return None
    l, u = lu_d

    sr, sc = A.shape
    if sr != sc:
        return None

    ud = u.to_dense()
    det = 1.0
    for r in range(sr):
        det *= ud[r][r]
    return det

