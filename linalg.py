from CSC import CSCMatrix
from CSR import CSRMatrix
from type1 import Vector, DenseMatrix
from typing import Tuple, Optional


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    Ожидается, что матрица L хранит единицы на главной диагонали.
    """
    A.shape[0] = A.shape[0]
    
    l_data, l_indices, l_indptr = [], [], [0]
    u_data, u_indices, u_indptr = [], [], [0]

    spa = [0] * A.shape[0]
    occupied = [False] * A.shape[0]

    for j in range(A.shape[0]):
        curr_occupied = []
        for k in range(A.indptr[j], A.indptr[j+1]):
            row = A.indices[k]
            spa[row] = A.data[k]
            occupied[row] = True
            curr_occupied.append(row)

        for i in range(j):
            if spa[i] != 0:
                pivot = spa[i]
                for k in range(l_indptr[i], l_indptr[i+1]):
                    row_l = l_indices[k]
                    spa[row_l] -= pivot * l_data[k]

        if spa[j] == 0:
            return None

        for i in range(j + 1):
            if spa[i] != 0:
                u_data.append(spa[i])
                u_indices.append(i)
                spa[i] = 0

        u_diag = u_data[-1]

        for i in range(j + 1, A.shape[0]):
            if spa[i] != 0:
                l_data.append(spa[i] / u_diag)
                l_indices.append(i)
                spa[i] = 0
        
        u_indptr.append(len(u_data))
        l_indptr.append(len(l_data))

    return (
        CSCMatrix(l_data, l_indices, l_indptr, (A.shape[0], A.shape[0])),
        CSCMatrix(u_data, u_indices, u_indptr, (A.shape[0], A.shape[0]))
    )


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    l, u = lu_decomposition(A)
    y: Vector = list(b)
    x: Vector = [0] * u.shape[0]

    for i in range(l.shape[0]):
        for k in range(l.indptr[i], l.indptr[i + 1]):
            row = l.indices[k]
            if row > i:
                y[row] -= l.data[k] * y[i]

    x = y
    for i in range(u.shape[0] - 1, -1, -1):
        elem_to_devide = 0
        for k in range(u.indptr[i], u.indptr[i + 1]):
            row = u.indices[k]
            if row == i:
                elem_to_devide = u.data[k]

        if elem_to_devide == 0:
            return None
        
        x[i] /= elem_to_devide

        for k in range(u.indptr[i], u.indptr[i + 1]):
            row = u.indices[k]
            if row < i:
                x[row] -= u.data[k] * x[i]

    return x


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    l, u = lu_decomposition(A)

    det_l = 1
    det_u = 1
    for i in range(u.shape[0]):
        for k in range(u.indptr[i], u.indptr[i + 1]):
            row = u.indices[k]
            if row == i:
                det_u *= u.data[k]

    return det_u * det_l
