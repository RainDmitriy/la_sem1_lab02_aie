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

    row_adj = [dict() for _ in range(n)]
    col_adj = [dict() for _ in range(n)]

    for c in range(m):
        for k in range(A.indptr[c], A.indptr[c + 1]):
            r = A.indices[k]
            val = A.data[k]
            row_adj[r][c] = val
            col_adj[c][r] = val

    L_triplets = []
    U_triplets = []

    for k in range(n):
        pivot = col_adj[k].get(k, 0.0)

        if pivot == 0:
            return None

        U_triplets.append((k, k, pivot))
        L_triplets.append((k, k, 1.0))

        u_indices = []
        u_vals = []

        if row_adj[k]:
            for c, val in sorted(row_adj[k].items()):
                if c > k:
                    if abs(val) > 1e-15:
                        U_triplets.append((k, c, val))
                        u_indices.append(c)
                        u_vals.append(val)

        l_indices = []
        l_vals = []

        if col_adj[k]:
            for r, val in sorted(col_adj[k].items()):
                if r > k:
                    l_val = val / pivot
                    if abs(l_val) > 1e-15:
                        L_triplets.append((r, k, l_val))
                        l_indices.append(r)
                        l_vals.append(l_val)

        for i in range(len(l_indices)):
            r = l_indices[i]
            l_val = l_vals[i]

            for j in range(len(u_indices)):
                c = u_indices[j]
                u_val = u_vals[j]

                update = l_val * u_val

                old_val = row_adj[r].get(c, 0.0)
                new_val = old_val - update

                if abs(new_val) > 1e-15:
                    row_adj[r][c] = new_val
                    col_adj[c][r] = new_val
                elif c in row_adj[r]:
                    del row_adj[r][c]
                    del col_adj[c][r]

    def to_csc(triplets, size):
        triplets.sort(key=lambda x: (x[1], x[0]))

        data = [x[2] for x in triplets]
        indices = [x[0] for x in triplets]
        indptr = [0] * (size + 1)

        curr_col = 0
        for _, col, _ in triplets:
            while curr_col < col:
                curr_col += 1
                indptr[curr_col + 1] = indptr[curr_col]
            indptr[curr_col + 1] += 1

        while curr_col < size - 1:
            curr_col += 1
            indptr[curr_col + 1] = indptr[curr_col]

        return CSCMatrix(data, indices, indptr, (size, size))

    return to_csc(L_triplets, n), to_csc(U_triplets, n)


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    lu = lu_decomposition(A)
    if lu is None:
        return None
    L, U = lu
    n = len(b)

    y = list(b)
    for j in range(n):
        if y[j] != 0:
            for k in range(L.indptr[j], L.indptr[j + 1]):
                row = L.indices[k]
                val = L.data[k]
                if row > j:
                    y[row] -= val * y[j]

    x = list(y)
    for j in range(n - 1, -1, -1):
        diag_val = 0
        for k in range(U.indptr[j], U.indptr[j + 1]):
            if U.indices[k] == j:
                diag_val = U.data[k]
                break

        if diag_val == 0: return None

        x[j] /= diag_val

        if x[j] != 0:
            for k in range(U.indptr[j], U.indptr[j + 1]):
                row = U.indices[k]
                val = U.data[k]
                if row < j:
                    x[row] -= val * x[j]

    return x


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    lu = lu_decomposition(A)
    if lu is None:
        return 0.0

    _, U = lu
    det = 1
    n = U.shape[0]

    for j in range(n):
        diag_found = False
        for k in range(U.indptr[j], U.indptr[j + 1]):
            if U.indices[k] == j:
                det *= U.data[k]
                diag_found = True
                break
        if not diag_found:
            return 0.0

    return det

