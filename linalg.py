from CSC import CSCMatrix
from CSR import CSRMatrix
from matrix_types import Vector
from typing import Tuple, Optional, List, Dict, Set

ZERO_TOL = 1e-12


def _build_symmetric_pattern(A: CSCMatrix) -> List[Set[int]]:
    n = A.shape[0]
    graph = [set() for _ in range(n)]
    for j in range(n):
        start = A.indptr[j]
        end = A.indptr[j + 1]
        for idx in range(start, end):
            i = A.indices[idx]
            if i != j:
                graph[i].add(j)
                graph[j].add(i)
    return graph


def _approximate_minimum_degree_ordering(A: CSCMatrix) -> List[int]:
    n = A.shape[0]
    graph = _build_symmetric_pattern(A)
    degree = [len(graph[i]) for i in range(n)]
    eliminated = [False] * n
    permutation = []

    for _ in range(n):
        min_deg = float('inf')
        v = -1
        for i in range(n):
            if not eliminated[i] and degree[i] < min_deg:
                min_deg = degree[i]
                v = i
        if v == -1:
            break
        eliminated[v] = True
        permutation.append(v)

        for u in list(graph[v]):
            if not eliminated[u]:
                graph[u].discard(v)
                degree[u] -= 1

    return permutation


def _apply_permutation_to_csc(A: CSCMatrix, perm: List[int]) -> CSCMatrix:
    n = A.shape[0]
    inv_perm = [0] * n
    for i, p in enumerate(perm):
        inv_perm[p] = i

    new_cols: List[Dict[int, float]] = [{} for _ in range(n)]
    for old_j in range(n):
        new_j = inv_perm[old_j]
        start = A.indptr[old_j]
        end = A.indptr[old_j + 1]
        for idx in range(start, end):
            old_i = A.indices[idx]
            new_i = inv_perm[old_i]
            val = A.data[idx]
            if abs(val) > ZERO_TOL:
                new_cols[new_j][new_i] = val

    data, indices, indptr = [], [], [0]
    for j in range(n):
        for i in sorted(new_cols[j].keys()):
            data.append(new_cols[j][i])
            indices.append(i)
        indptr.append(len(data))

    return CSCMatrix(data, indices, indptr, (n, n))


def _undo_permutation(x_perm: Vector, perm: List[int]) -> Vector:
    n = len(x_perm)
    x = [0.0] * n
    for i in range(n):
        x[perm[i]] = x_perm[i]
    return x


def _lu_no_pivot(A: CSCMatrix) -> Optional[Tuple[CSRMatrix, CSCMatrix]]:
    n = A.shape[0]
    A_cols: List[Dict[int, float]] = []
    for j in range(n):
        col = {}
        start = A.indptr[j]
        end = A.indptr[j + 1]
        for idx in range(start, end):
            i = A.indices[idx]
            val = A.data[idx]
            if abs(val) > ZERO_TOL:
                col[i] = val
        A_cols.append(col)

    L_rows: List[Dict[int, float]] = [{} for _ in range(n)]
    U_cols: List[Dict[int, float]] = [{} for _ in range(n)]

    for j in range(n):
        y = A_cols[j].copy()
        for i in sorted(y.keys()):
            if i >= n or abs(y[i]) <= ZERO_TOL:
                continue
            sum_val = 0.0
            for k, L_ik in L_rows[i].items():
                if k in y:
                    sum_val += L_ik * y[k]
            y[i] -= sum_val

        if j not in y or abs(y[j]) <= ZERO_TOL:
            return None
        u_jj = y[j]

        for i in list(y.keys()):
            if i <= j and abs(y[i]) > ZERO_TOL:
                U_cols[j][i] = y[i]
        for i in list(y.keys()):
            if i > j and abs(y[i]) > ZERO_TOL:
                L_ij = y[i] / u_jj
                if abs(L_ij) > ZERO_TOL:
                    L_rows[i][j] = L_ij

    L_data, L_indices, L_indptr = [], [], [0]
    for i in range(n):
        cols = sorted([j for j in L_rows[i].keys() if j < i])
        for j in cols:
            L_data.append(L_rows[i][j])
            L_indices.append(j)
        L_indptr.append(len(L_data))
    L_csr = CSRMatrix(L_data, L_indices, L_indptr, (n, n))

    U_data, U_indices, U_indptr = [], [], [0]
    for j in range(n):
        rows = sorted(U_cols[j].keys())
        for i in rows:
            if i <= j:
                val = U_cols[j][i]
                if abs(val) > ZERO_TOL:
                    U_data.append(val)
                    U_indices.append(i)
        U_indptr.append(len(U_data))
    U_csc = CSCMatrix(U_data, U_indices, U_indptr, (n, n))

    return L_csr, U_csc


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSRMatrix, CSCMatrix]]:
    if A.shape[0] != A.shape[1]:
        return None
    perm = _approximate_minimum_degree_ordering(A)
    A_perm = _apply_permutation_to_csc(A, perm)
    return _lu_no_pivot(A_perm)


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    if A.shape[0] != len(b):
        return None
    perm = _approximate_minimum_degree_ordering(A)
    A_perm = _apply_permutation_to_csc(A, perm)
    b_perm = [b[perm[i]] for i in range(len(b))]
    lu = _lu_no_pivot(A_perm)
    if lu is None:
        return None
    L, U = lu
    n = len(b)

    y = [0.0] * n
    for i in range(n):
        sum_val = 0.0
        start = L.indptr[i]
        end = L.indptr[i + 1]
        for idx in range(start, end):
            j = L.indices[idx]
            sum_val += L.data[idx] * y[j]
        y[i] = b_perm[i] - sum_val

    x_perm = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_val = 0.0
        start = U.indptr[i]
        end = U.indptr[i + 1]
        u_ii = 0.0
        found = False
        for idx in range(start, end):
            row = U.indices[idx]
            if row == i:
                u_ii = U.data[idx]
                found = True
            elif row > i:
                sum_val += U.data[idx] * x_perm[row]
        if not found or abs(u_ii) < ZERO_TOL:
            return None
        x_perm[i] = (y[i] - sum_val) / u_ii

    return _undo_permutation(x_perm, perm)


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    if A.shape[0] != A.shape[1]:
        return None
    perm = _approximate_minimum_degree_ordering(A)
    A_perm = _apply_permutation_to_csc(A, perm)
    lu = _lu_no_pivot(A_perm)
    if lu is None:
        return None
    _, U = lu
    det = 1.0
    n = U.shape[0]
    for i in range(n):
        u_ii = 0.0
        start = U.indptr[i]
        end = U.indptr[i + 1]
        found = False
        for idx in range(start, end):
            if U.indices[idx] == i:
                u_ii = U.data[idx]
                found = True
                break
        if not found or abs(u_ii) < ZERO_TOL:
            return 0.0
        det *= u_ii
        if abs(det) < 1e-300:
            break
    return det