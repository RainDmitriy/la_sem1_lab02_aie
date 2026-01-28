from CSC import CSCMatrix
from type import Vector
from typing import Tuple, Optional
EPS = 1e-10


def lu_decomposition(mat: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы
    Возвращает пару (L, U), где L – нижняя треугольная с единицами на диагонали,
    U – верхняя треугольная матрица.
    """
    if mat.shape[0] != mat.shape[1]:
        raise ValueError("LU-разложение определено только для квадратных матриц")




    sr = mat.shape[0]
    if mat.nnz == 0:
        L = [[0.0] * sr for _ in range(sr)]
        U = [[0.0] * sr for _ in range(sr)]
        for i in range(sr):
            L[i][i] = 1.0
        return CSCMatrix.from_dense(L), CSCMatrix.from_dense(U)
    Lc = [{} for _ in range(sr)]  # L по столбцам (CSC формат)
    Ur = [{} for _ in range(sr)]  # U по строкам
    rows = [{} for _ in range(sr)]
    for c in range(sr):
        for p in range(mat.indptr[c], mat.indptr[c + 1]):
            r = mat.indices[p]
            rows[r][c] = float(mat.data[p])
    upd = [{} for _ in range(sr)]

    for r in range(sr):
        row = {}
        for c, d in rows[r].items():
            if c >= r:
                row[c] = float(d)
        for c, d in upd[r].items():
            if c >= r:
                row[c] = row.get(c, 0.0) + float(d)
        p = row.get(r, 0.0)
        if abs(p) < EPS:
            return None
        Ur[r] = {}
        for c, d in row.items():
            if c >= r and abs(d) > EPS:
                Ur[r][c] = d
        Ur[r][r] = p
        Lc[r][r] = 1.0
        for r2 in range(r + 1, sr):
            elem = 0.0
            if r in rows[r2]:
                elem += float(rows[r2][r])
            if r in upd[r2]:
                elem += float(upd[r2][r])
            if abs(elem) > EPS:
                f = elem / p
                Lc[r][r2] = f
                for c, du in Ur[r].items():
                    if c > r:
                        d = -f * du
                        if abs(d) > EPS:
                            upd[r2][c] = upd[r2].get(c, 0.0) + d

    L_data, L_indices, L_indptr = [], [], [0]
    for c in range(sr):
        rr = sorted(Lc[c].keys())
        for r in rr:
            L_data.append(Lc[c][r])
            L_indices.append(r)
        L_indptr.append(len(L_data))
    Uc = [{} for _ in range(sr)]
    for r in range(sr):
        for c, d in Ur[r].items():
            Uc[c][r] = d
    U_data, U_indices, U_indptr = [], [], [0]
    for c in range(sr):
        rr = sorted(Uc[c].keys())
        for r in rr:
            U_data.append(Uc[c][r])
            U_indices.append(r)
        U_indptr.append(len(U_data))
    return (CSCMatrix(L_data, L_indices, L_indptr, (sr, sr)),
            CSCMatrix(U_data, U_indices, U_indptr, (sr, sr)))


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение
    """
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None
    L, U = lu_result
    n = len(b)
    L_dense = L.to_dense()
    U_dense = U.to_dense()
    y = [0.0] * n
    for i in range(n):
        sum_val = 0.0
        for j in range(i):
            sum_val += L_dense[i][j] * y[j]
        y[i] = b[i] - sum_val  # L[i][i] = 1.0

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_val = 0.0
        for j in range(i + 1, n):
            sum_val += U_dense[i][j] * x[j]
        if abs(U_dense[i][i]) < EPS:
            return None
        x[i] = (y[i] - sum_val) / U_dense[i][i]

    return x

def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение
    det(A) = det(L) * det(U)
    """
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None
    _, U = lu_result
    n = A.shape[0]
    det = 1.0
    dense_U = U.to_dense()
    for i in range(n):
        det *= dense_U[i][i]

    return det
