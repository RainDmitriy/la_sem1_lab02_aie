from CSC import CSCMatrix
from type import Vector
from typing import Tuple, Optional

EPS = 1e-10


def lu_decomposition(mat: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    Выполняет LU‑разложение матрицы в формате CSC.
    Возвращает пару (L, U), где L – нижняя треугольная с единицами на диагонали,
    U – верхняя треугольная матрица.
    """
    sr = mat.shape[0]

    Lc = [{} for _ in range(sr)]
    Ur = [{} for _ in range(sr)]

    # преобразуем входную матрицу в словарь строк для быстрого доступа
    rows = [{} for _ in range(sr)]
    for c in range(sr):
        for p in range(mat.indptr[c], mat.indptr[c + 1]):
            r = mat.indices[p]
            rows[r][c] = float(mat.data[p])

    # массив для накопления обновлений
    upd = [{} for _ in range(sr)]

    for r in range(sr):
        row = {}

        # собираем элементы из исходной матрицы
        for c, d in rows[r].items():
            if c >= r:
                row[c] = float(d)

        # добавляем накопленные обновления
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

    # преобразуем L в CSC
    L_data, L_indices, L_indptr = [], [], [0]
    for c in range(sr):
        rr = sorted(Lc[c].keys())
        for r in rr:
            L_data.append(Lc[c][r])
            L_indices.append(r)
        L_indptr.append(len(L_data))

    # преобразуем U в CSC (транспонируем словарь строк)
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
    Решает систему линейных уравнений Ax = b с использованием LU‑разложения.
    """
    lu = lu_decomposition(A)
    if lu is None:
        return None

    L, U = lu
    sr = A.shape[0]

    ld = L.to_dense()
    ud = U.to_dense()

    # прямой ход: Ly = b
    y = [0.0] * sr
    for r in range(sr):
        s = 0.0
        for c in range(r):
            s += ld[r][c] * y[c]
        y[r] = b[r] - s

    # обратный ход: Ux = y
    x = [0.0] * sr
    for r in range(sr - 1, -1, -1):
        s = 0.0
        for c in range(r + 1, sr):
            s += ud[r][c] * x[c]

        if abs(ud[r][r]) < EPS:
            return None

        x[r] = (y[r] - s) / ud[r][r]

    return x


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Вычисляет определитель матрицы с помощью LU‑разложения.
    """
    lu = lu_decomposition(A)
    if lu is None:
        return None

    _, U = lu
    sr = A.shape[0]

    ud = U.to_dense()
    det = 1.0
    for r in range(sr):
        det *= ud[r][r]

    return det