from CSC import CSCMatrix
from type import Vector
from typing import Tuple, Optional
from collections import defaultdict
from COO import COOMatrix


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы БЕЗ преобразования в dense.
    Использует алгоритм Гаусса с учётом разреженности.
    """
    n = A.shape[0]

    if n != A.shape[1]:
        raise ValueError("Матрица должна быть квадратной")

    rows = [[] for _ in range(n)]

    for j in range(n):
        start, end = A.indptr[j], A.indptr[j + 1]
        for pos in range(start, end):
            i = A.indices[pos]
            val = A.data[pos]
            rows[i].append((j, val))

    for i in range(n):
        rows[i].sort(key=lambda x: x[0])
        rows[i] = dict(rows[i])

    L_dict = defaultdict(float)
    U_dict = defaultdict(float)

    current_row = [0.0] * n

    for k in range(n):
        for col, val in rows[k].items():
            current_row[col] = val

        for j in range(k):
            if (k, j) in L_dict:
                factor = L_dict[(k, j)]
                for col, u_val in U_dict.items():
                    if col[0] == j and col[1] >= j:
                        current_row[col[1]] -= factor * u_val

        diag = current_row[k]
        if abs(diag) < 1e-12:
            return None

        U_dict[(k, k)] = diag

        for j in range(k + 1, n):
            val = current_row[j]
            if abs(val) > 1e-12:
                if j < n:
                    U_dict[(k, j)] = val
                L_val = val / diag
                if abs(L_val) > 1e-12:
                    L_dict[(j, k)] = L_val

        current_row = [0.0] * n

    for i in range(n):
        L_dict[(i, i)] = 1.0

    L_data, L_rows, L_cols = [], [], []
    U_data, U_rows, U_cols = [], [], []

    for (i, j), val in L_dict.items():
        if abs(val) > 1e-12:
            L_data.append(val)
            L_rows.append(i)
            L_cols.append(j)

    for (i, j), val in U_dict.items():
        if abs(val) > 1e-12:
            U_data.append(val)
            U_rows.append(i)
            U_cols.append(j)

    L_coo = COOMatrix(L_data, L_rows, L_cols, (n, n))
    U_coo = COOMatrix(U_data, U_rows, U_cols, (n, n))

    return L_coo._to_csc(), U_coo._to_csc()


def solve_triangular_sparse(L: CSCMatrix, U: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение треугольных систем Ly = b и Ux = y для разреженных матриц.
    """
    n = len(b)

    y = [0.0] * n
    for i in range(n):
        total = b[i]
        start, end = L.indptr[i], L.indptr[i + 1]
        for pos in range(start, end):
            row = L.indices[pos]
            if row < i:
                total -= L.data[pos] * y[row]
        y[i] = total

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        total = y[i]
        for j in range(i + 1, n):
            start, end = U.indptr[j], U.indptr[j + 1]
            left, right = start, end - 1
            while left <= right:
                mid = (left + right) // 2
                row = U.indices[mid]
                if row == i:
                    total -= U.data[mid] * x[j]
                    break
                elif row < i:
                    left = mid + 1
                else:
                    right = mid - 1

        start, end = U.indptr[i], U.indptr[i + 1]
        left, right = start, end - 1
        diag = None
        while left <= right:
            mid = (left + right) // 2
            row = U.indices[mid]
            if row == i:
                diag = U.data[mid]
                break
            elif row < i:
                left = mid + 1
            else:
                right = mid - 1

        if diag is None or abs(diag) < 1e-12:
            return None

        x[i] = total / diag

    return x


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение в разреженном формате.
    """
    # Получаем LU-разложение
    result = lu_decomposition(A)
    if result is None:
        return None

    L, U = result
    n = A.shape[0]

    if len(b) != n:
        raise ValueError("Размер вектора b не совпадает с размером матрицы A")

    return solve_triangular_sparse(L, U, b)


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение в разреженном формате.
    det(A) = произведение диагональных элементов U.
    """
    result = lu_decomposition(A)
    if result is None:
        return None

    L, U = result
    n = A.shape[0]
    det = 1.0
    for i in range(n):
        # Ищем диагональный элемент U[i][i] в CSC формате
        start, end = U.indptr[i], U.indptr[i + 1]
        # Бинарный поиск
        left, right = start, end - 1
        found = False
        while left <= right:
            mid = (left + right) // 2
            row = U.indices[mid]
            if row == i:
                det *= U.data[mid]
                found = True
                break
            elif row < i:
                left = mid + 1
            else:
                right = mid - 1

        if not found:
            return 0.0

    return det


def lu_decomposition_simple(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    Упрощённая версия LU-разложения для тестирования.
    """
    from COO import COOMatrix
    coo = A._to_coo()

    n = A.shape[0]

    L_dense = [[0.0] * n for _ in range(n)]
    U_dense = [[0.0] * n for _ in range(n)]

    dense_A = [[0.0] * n for _ in range(n)]
    for i in range(len(coo.data)):
        r, c, val = coo.row[i], coo.col[i], coo.data[i]
        dense_A[r][c] = val

    try:
        for i in range(n):
            # U
            for j in range(i, n):
                total = dense_A[i][j]
                for k in range(i):
                    total -= L_dense[i][k] * U_dense[k][j]
                U_dense[i][j] = total

            L_dense[i][i] = 1.0
            for j in range(i + 1, n):
                total = dense_A[j][i]
                for k in range(i):
                    total -= L_dense[j][k] * U_dense[k][i]
                if abs(U_dense[i][i]) < 1e-12:
                    return None
                L_dense[j][i] = total / U_dense[i][i]
    except ZeroDivisionError:
        return None

    L_coo = COOMatrix.from_dense(L_dense)
    U_coo = COOMatrix.from_dense(U_dense)

    return L_coo._to_csc(), U_coo._to_csc()