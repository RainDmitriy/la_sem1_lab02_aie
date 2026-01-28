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
    n = A.shape[0]

    U_data = A.data[:]
    U_indices = A.indices[:]
    U_indptr = A.indptr[:]

    L_data = []
    L_indices = []
    L_indptr = [0] * (n + 1)

    for k in range(n):
        col_start = U_indptr[k]
        col_end = U_indptr[k + 1]
        diag_pos = -1

        for pos in range(col_start, col_end):
            if U_indices[pos] == k:
                diag_pos = pos
                break

        if diag_pos == -1 or abs(U_data[diag_pos]) < 1e-14:
            return None

        u_kk = U_data[diag_pos]

        for pos in range(diag_pos + 1, col_end):
            row_idx = U_indices[pos]
            if row_idx > k:
                L_data.append(U_data[pos] / u_kk)
                L_indices.append(row_idx)

        L_indptr[k + 1] = len(L_data)

        for j in range(k + 1, n):
            col_j_start = U_indptr[j]
            col_j_end = U_indptr[j + 1]
            #столбец k и столбец j
            new_data = []
            new_indices = []
            p1 = col_start
            p2 = col_j_start
            while p1 < col_end and p2 < col_j_end:
                idx1, idx2 = U_indices[p1], U_indices[p2]

                if idx1 == idx2:
                    if idx1 > k:
                        l_ki = U_data[p1] / u_kk  # коэффициент
                        new_data.append(U_data[p2] - l_ki * U_data[diag_pos])
                    p1 += 1
                    p2 += 1
                elif idx1 < idx2:
                    if idx1 > k:
                        new_data.append(U_data[p1])
                        new_indices.append(idx1)
                    p1 += 1
                else:
                    new_data.append(U_data[p2])
                    new_indices.append(idx2)
                    p2 += 1

            while p1 < col_end:
                if U_indices[p1] > k:
                    new_data.append(U_data[p1])
                    new_indices.append(U_indices[p1])
                p1 += 1
            while p2 < col_j_end:
                new_data.append(U_data[p2])
                new_indices.append(U_indices[p2])
                p2 += 1

            for pos in range(len(new_data)):
                U_data[col_j_start + pos] = new_data[pos]
                U_indices[col_j_start + pos] = new_indices[pos]

    L_data_full = [1.0] * n
    L_indices_full = list(range(n))
    L_indptr_full = [0] * (n + 1)
    for j in range(n):
        L_data_full.append(1.0)
        L_indices_full.append(j)
        for pos in range(L_indptr[j], L_indptr[j + 1]):
            L_data_full.append(L_data[pos])
            L_indices_full.append(L_indices[pos])
        L_indptr_full[j + 1] = len(L_data_full)
    L = CSCMatrix(L_data_full, L_indices_full, L_indptr_full, (n, n))
    U = CSCMatrix(U_data, U_indices, U_indptr, (n, n))

    return L, U

def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    LU = lu_decomposition(A)
    if LU is None:
        raise ValueError("LU-разложение не удалось выполнить")

    L, U = LU
    n = len(b)

    y = [0.0] * n
    for i in range(n):
        s = b[i]
        row_start = L.indptr[i]
        row_end = L.indptr[i + 1]
        for k in range(row_start, row_end):
            col_idx = L.indices[k]
            if col_idx < i:
                s -= L.data[k] * y[col_idx]
        y[i] = s

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = y[i]
        row_start = U.indptr[i]
        row_end = U.indptr[i + 1]
        for k in range(row_start + 1, row_end):
            col_idx = U.indices[k]
            s -= U.data[k] * x[col_idx]
        diag_pos = row_start
        while diag_pos < row_end and U.indices[diag_pos] != i:
            diag_pos += 1
        if diag_pos < row_end:
            x[i] = s / U.data[diag_pos]
        else:
            raise ValueError(f"Диагональный элемент U[{i},{i}] не найден в строке {i}")

    return x

def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    LU = lu_decomposition(A)
    if LU is None:
        raise ValueError("LU-разложение не удалось, матрица вырождена ")

    L, U = LU
    det = 1.0
    n = U.shape[0]
    found_diag = True

    for i in range(n):
        row_start = U.indptr[i]
        row_end = U.indptr[i + 1]
        diag_found = False
        for k in range(row_start, row_end):
            if U.indices[k] == i:
                det *= U.data[k]
                diag_found = True
                break
        if not diag_found:
            found_diag = False
            break
    if not found_diag:
        raise ValueError(f"Диагональный элемент U[{i},{i}] отсутствует")

    return det

