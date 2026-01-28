from CSC import CSCMatrix
from COO import COOMatrix
from CSR import CSRMatrix
from type import Vector
from typing import Tuple, Optional, Dict, List

TOLERANCE = 1e-12

def _add_dict(dict: Dict[int, float], key: int, delta: float) -> None:
    """добавление в словарь"""
    new_val = dict.get(key, 0.0) + delta
    if abs(new_val) < TOLERANCE:
        dict.pop(key, None)
        return
    dict[key] = new_val


def _set_val(
        rows: Dict[int, Dict[int, float]],
        cols: Dict[int, Dict[int, float]],
        i: int,
        j: int,
        val: float) -> None:
    """значения в матрице"""
    if abs(val) < TOLERANCE:
        r = rows.get(i)
        if r is not None and j in r:
            del r[j]
            if not r:
                del rows[i]
        c = cols.get(j)
        if c is not None and i in c:
            del c[i]
            if not c:
                del cols[j]
        return
    r = rows.get(i)
    if r is None:
        r = {}
        rows[i] = r
    r[j] = val
    c = cols.get(j)
    if c is None:
        c = {}
        cols[j] = c
    c[i] = val


def _new_structure(A: CSCMatrix) -> Tuple[Dict[int, Dict[int, float]], Dict[int, Dict[int, float]]]:
    """обеспечение быстрого доступа по строкам и по столбцам."""
    n_rows, n_cols = A.shape
    rows: Dict[int, Dict[int, float]] = {}
    cols: Dict[int, Dict[int, float]] = {}

    for j in range(n_cols):
        start = A.indptr[j]
        end = A.indptr[j + 1]
        for p in range(start, end):
            i = int(A.indices[p])
            v = float(A.data[p])
            if abs(v) < TOLERANCE:
                continue
            r = rows.get(i)
            if r is None:
                r = {}
                rows[i] = r
            _add_dict(r, j, v)


            c = cols.get(j)
            if c is None:
                c = {}
                cols[j] = c
            _add_dict(c, i, v)

    for row_index, row_dict in list(rows.items()):
        if not row_dict:
            del rows[row_index]
    for col_index, col_dict in list(cols.items()):
        if not col_dict:
            del cols[col_index]

    return rows, cols


def _triplets_to_csc(n_rows: int, n_cols: int, triplets: List[Tuple[int, int, float]]) -> CSCMatrix:
    """список (row, col, value) преобразовывается в CSC матрицу."""
    triplets.sort(key=lambda x: (x[1], x[0]))
    data: List[float] = []
    indices: List[int] = []
    indptr: List[int] = [0]

    current_col = 0
    for row, col, val in triplets:
        if abs(val) < TOLERANCE:
            continue
        while current_col < col:
            indptr.append(len(data))
            current_col += 1
        data.append(val)
        indices.append(row)

    while current_col < n_cols:
        indptr.append(len(data))
        current_col += 1

    if len(indptr) == n_cols:
        indptr.append(len(data))

    return CSCMatrix(data, indices, indptr, (n_rows, n_cols))


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    Ожидается, что матрица L хранит единицы на главной диагонали.
    """
    n_rows, n_cols = A.shape
    if n_rows != n_cols:
        return None
    n = n_rows

    if n == 0:
        empty = CSCMatrix([], [], [0], (0, 0))
        return empty, empty
    rows, cols = _new_structure(A)

    L_trip: List[Tuple[int, int, float]] = []
    U_trip: List[Tuple[int, int, float]] = []

    for i in range(n):
        L_trip.append((i, i, 1.0))

    for k in range(n):
        row_k = rows.get(k, {})

        for j in range(k, n):
            total = row_k.get(j, 0.0)
            for t in range(k):
                l_kt = 0.0
                for (r, c, v) in L_trip:
                    if r == k and c == t:
                        l_kt = v
                        break

                u_tj = 0.0
                for (r, c, v) in U_trip:
                    if r == t and c == j:
                        u_tj = v
                        break
                total -= l_kt * u_tj

            if abs(total) > TOLERANCE:
                U_trip.append((k, j, float(total)))
        pivot = 0.0
        for (r, c, v) in U_trip:
            if r == k and c == k:
                pivot = v
                break
        if abs(pivot) < TOLERANCE:
            return None
        col_k = cols.get(k, {})

        for i in range(k + 1, n):
            if i in col_k:
                total = col_k[i]
                for t in range(k):
                    l_it = 0.0
                    for (r, c, v) in L_trip:
                        if r == i and c == t:
                            l_it = v
                            break

                    u_tk = 0.0
                    for (r, c, v) in U_trip:
                        if r == t and c == k:
                            u_tk = v
                            break
                    total -= l_it * u_tk
                l_ik = total / pivot
                if abs(l_ik) > TOLERANCE:
                    L_trip.append((i, k, float(l_ik)))
        for i in range(k + 1, n):
            l_ik = 0.0
            for (r, c, v) in L_trip:
                if r == i and c == k:
                    l_ik = v
                    break

            if abs(l_ik) < TOLERANCE:
                continue

            for (r, c, u_kj) in U_trip:
                if r == k and c > k:
                    current = rows.get(i, {}).get(c, 0.0)
                    new_val = current - l_ik * u_kj
                    _set_val(rows, cols, i, c, new_val)

    L_csc = _triplets_to_csc(n, n, L_trip)
    U_csc = _triplets_to_csc(n, n, U_trip)

    return L_csc, U_csc

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
    """решение U*x = y"""
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
    if any(val != val for val in x):
        return None
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
