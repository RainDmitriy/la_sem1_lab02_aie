from CSC import CSCMatrix
from CSR import CSRMatrix
from type import Vector
from typing import Tuple, Optional
EPS = 1e-10
def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    Ожидается, что матрица L хранит единицы на главной диагонали.
    """
    sr = mat.shape[0]
    Lc = [{} for _ in range(sr)]
    Ur = [{} for _ in range(sr)]
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
    Решение СЛАУ Ax = b через LU-разложение.
    """
    lu = lu_decomposition(A)
    if lu is None:
        return None
    L, U = lu
    sr = A.shape[0]
    y = [0.0] * sr
    for i in range(sr):
        s = b[i]
        col_start = L.indptr[i]
        col_end = L.indptr[i + 1]
        for k in range(col_start, col_end):
            if L.indices[k] < i:
                s -= L.data[k] * y[L.indices[k]]
        y[i] = s
    x = [0.0] * sr
    for i in range(sr - 1, -1, -1):
        s = y[i]
        col_start = U.indptr[i]
        col_end = U.indptr[i + 1]
        diag_pos = -1
        for pos in range(col_start, col_end):
            if U.indices[pos] == i:
                diag_pos = pos
                break
        if diag_pos == -1 or abs(U.data[diag_pos]) < EPS:
            return None
        for pos in range(diag_pos + 1, col_end):
            s -= U.data[pos] * x[U.indices[pos]]
        x[i] = s / U.data[diag_pos]
    return x

def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    lu = lu_decomposition(A)
    if lu is None:
        return None
    L, U = lu
    sr = U.shape[0]
    det = 1.0
    for i in range(sr):
        col_start = U.indptr[i]
        col_end = U.indptr[i + 1]
        diag_pos = -1
        for pos in range(col_start, col_end):
            if U.indices[pos] == i:
                diag_pos = pos
                break
        if diag_pos == -1 or abs(U.data[diag_pos]) < EPS:
            return 0.0
        det *= U.data[diag_pos]
    return det