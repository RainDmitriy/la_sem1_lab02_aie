from CSC import CSCMatrix
from CSR import CSRMatrix
from types import Vector
from typing import Tuple, Optional


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    Ожидается, что матрица L хранит единицы на главной диагонали.
    """
    n, m = A.shape

    a_dict = dict()
    for c in range(m):
        for k in range(A.indptr[c], A.indptr[c + 1]):
            a_dict[(A.indices[k], c)] = A.data[k]

    l_dict, u_dict = dict(), dict()

    for i in range(n):
        for k in range(i, n):
            s_sum = sum(l_dict.get((i, s), 0) * u_dict.get((s, k), 0) for s in range(i))
            val = a_dict.get((i, k), 0) - s_sum
            u_dict[(i, k)] = val

        diag_u = u_dict.get((i, i), 0)
        if diag_u == 0:
            return None

        l_dict[(i, i)] = 1.0
        for k in range(i + 1, n):
            s_sum = sum(l_dict.get((k, s), 0) * u_dict.get((s, i), 0) for s in range(i))
            val = (a_dict.get((k, i), 0) - s_sum) / diag_u
            l_dict[(k, i)] = val

    def to_csc(data_dict):
        sorted_items = sorted(data_dict.items(), key=lambda x: (x[0][1], x[0][0]))

        data = [v for k, v in sorted_items]
        indices = [k[0] for k, v in sorted_items]

        indptr = [0] * (n + 1)
        for (r, c), v in sorted_items:
            indptr[c + 1] += 1

        for i in range(n):
            indptr[i + 1] += indptr[i]

        return CSCMatrix(data, indices, indptr, (n, n))

    return to_csc(l_dict), to_csc(u_dict)


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
