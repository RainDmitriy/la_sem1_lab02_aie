from __future__ import annotations
from CSC import CSCMatrix
from CSR import CSRMatrix
from types import Vector
from typing import Tuple, Optional, Dict, List
_EPS = 1e-12

def _add_to_map(m: Dict[int, float], key: int, delta: float) -> None:
    new_val = m.get(key, 0.0) + delta
    if abs(new_val) < _EPS:
        m.pop(key, None)
        return
    m[key] = new_val

def _set_entry(
    rows: Dict[int, Dict[int, float]],
    cols: Dict[int, Dict[int, float]],
    i: int,
    j: int,
    value: float,
) -> None:
    if abs(value) < _EPS:
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
    r[j] = value
    c = cols.get(j)
    if c is None:
        c = {}
        cols[j] = c
    c[i] = value

def _build_row_col_maps(A: CSCMatrix) -> Tuple[Dict[int, Dict[int, float]], Dict[int, Dict[int, float]]]:
    n_rows, n_cols = A.shape
    rows: Dict[int, Dict[int, float]] = {}
    cols: Dict[int, Dict[int, float]] = {}
    for j in range(n_cols):
        start = A.indptr[j]
        end = A.indptr[j + 1]
        for p in range(start, end):
            i = int(A.indices[p])
            v = float(A.data[p])
            if abs(v) < _EPS:
                continue
            r = rows.get(i)
            if r is None:
                r = {}
                rows[i] = r
            _add_to_map(r, j, v)
            c = cols.get(j)
            if c is None:
                c = {}
                cols[j] = c
            _add_to_map(c, i, v)
    for i in [ii for ii, r in rows.items() if not r]:
        del rows[i]
    for j in [jj for jj, c in cols.items() if not c]:
        del cols[j]
    return rows, cols

def _triplets_to_csc(n_rows: int, n_cols: int, triplets: List[Tuple[int, int, float]]) -> CSCMatrix:
    col_maps: Dict[int, Dict[int, float]] = {}
    for r, c, v in triplets:
        if abs(v) < _EPS:
            continue
        cm = col_maps.get(c)
        if cm is None:
            cm = {}
            col_maps[c] = cm
        cm[r] = cm.get(r, 0.0) + float(v)
        if abs(cm[r]) < _EPS:
            del cm[r]
            if not cm:
                del col_maps[c]
    data: List[float] = []
    indices: List[int] = []
    indptr: List[int] = [0]
    for c in range(n_cols):
        cm = col_maps.get(c)
        if not cm:
            indptr.append(len(data))
            continue
        items = [(r, v) for r, v in cm.items() if abs(v) >= _EPS]
        items.sort(key=lambda t: t[0])
        for r, v in items:
            indices.append(int(r))
            data.append(float(v))
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
    rows, cols = _build_row_col_maps(A)
    l_triplets: List[Tuple[int, int, float]] = []
    u_triplets: List[Tuple[int, int, float]] = []
    for i in range(n):
        l_triplets.append((i, i, 1.0))
    for k in range(n):
        row_k = rows.get(k, {})
        pivot = float(row_k.get(k, 0.0))
        if abs(pivot) < _EPS:
            return None
        for j, v in row_k.items():
            if j >= k and abs(v) >= _EPS:
                u_triplets.append((k, j, float(v)))
        col_k = cols.get(k, {})
        below = [(i, float(v)) for i, v in col_k.items() if i > k and abs(v) >= _EPS]
        for i, a_ik in below:
            mult = a_ik / pivot
            if abs(mult) >= _EPS:
                l_triplets.append((i, k, mult))
            _set_entry(rows, cols, i, k, 0.0)
            for j, u_kj in row_k.items():
                if j <= k:
                    continue
                a_ij = rows.get(i, {}).get(j, 0.0)
                new_val = float(a_ij) - mult * float(u_kj)
                _set_entry(rows, cols, i, j, new_val)
    return _triplets_to_csc(n, n, l_triplets), _triplets_to_csc(n, n, u_triplets)

def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    n_rows, n_cols = A.shape
    if n_rows != n_cols:
        return None
    n = n_rows
    if len(b) != n:
        return None
    if n == 0:
        return []
    rows, cols = _build_row_col_maps(A)
    l_cols: Dict[int, List[Tuple[int, float]]] = {}
    u_rows: Dict[int, Dict[int, float]] = {}
    for k in range(n):
        row_k = rows.get(k, {})
        pivot = float(row_k.get(k, 0.0))
        if abs(pivot) < _EPS:
            return None
        uk: Dict[int, float] = {}
        for j, v in row_k.items():
            if j >= k and abs(v) >= _EPS:
                uk[j] = float(v)
        u_rows[k] = uk
        col_k = cols.get(k, {})
        below = [(i, float(v)) for i, v in col_k.items() if i > k and abs(v) >= _EPS]
        for i, a_ik in below:
            mult = a_ik / pivot
            if abs(mult) >= _EPS:
                l_cols.setdefault(k, []).append((i, mult))
            _set_entry(rows, cols, i, k, 0.0)
            for j, u_kj in uk.items():
                if j == k:
                    continue
                a_ij = rows.get(i, {}).get(j, 0.0)
                new_val = float(a_ij) - mult * float(u_kj)
                _set_entry(rows, cols, i, j, new_val)
    y = [float(v) for v in b]
    for k in range(n):
        yk = y[k]
        for i, l_ik in l_cols.get(k, []):
            y[i] -= float(l_ik) * yk
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        row = u_rows.get(i, {})
        pivot = float(row.get(i, 0.0))
        if abs(pivot) < _EPS:
            return None
        s = 0.0
        for j, v in row.items():
            if j <= i:
                continue
            s += float(v) * x[j]
        x[i] = (y[i] - s) / pivot
    return x

def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    n_rows, n_cols = A.shape
    if n_rows != n_cols:
        return None
    n = n_rows
    if n == 0:
        return 1.0
    rows, cols = _build_row_col_maps(A)
    det = 1.0
    for k in range(n):
        row_k = rows.get(k, {})
        pivot = float(row_k.get(k, 0.0))
        if abs(pivot) < _EPS:
            return None
        det *= pivot
        col_k = cols.get(k, {})
        below = [(i, float(v)) for i, v in col_k.items() if i > k and abs(v) >= _EPS]
        for i, a_ik in below:
            mult = a_ik / pivot
            _set_entry(rows, cols, i, k, 0.0)
            for j, u_kj in row_k.items():
                if j == k:
                    continue
                a_ij = rows.get(i, {}).get(j, 0.0)
                new_val = float(a_ij) - mult * float(u_kj)
                _set_entry(rows, cols, i, j, new_val)
    return det