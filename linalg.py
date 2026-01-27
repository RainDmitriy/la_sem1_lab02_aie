from CSC import CSCMatrix
from CSR import CSRMatrix
from types1 import Vector, DenseMatrix
from typing import Tuple, Optional


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    Ожидается, что матрица L хранит единицы на главной диагонали.
    """
    l: DenseMatrix = [
        [1 if i == j else 0 for j in range(A.shape[1])] for i in range(A.shape[0])
    ]
    u: DenseMatrix = [[0 for _ in range(A.shape[1])] for _ in range(A.shape[0])]

    for i in range(A.shape[0]):

        for j in range(i, A.shape[1]):
            val_u = elem_from_csc(A, i, j)
            for k in range(i):
                val_u -= l[i][k] * u[k][j]
            u[i][j] = val_u

        for j in range(i + 1, A.shape[0]):
            if u[i][i] == 0:
                raise ValueError("нулевой главный минор")
            val_l = elem_from_csc(A, j, i)
            for k in range(i):
                val_l -= l[j][k] * u[k][i]
            l[j][i] = val_l / u[i][i]

    mat_l = CSCMatrix.from_dense(l)
    mat_u = CSCMatrix.from_dense(u)

    return mat_l, mat_u


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    l, u = lu_decomposition(A)
    y: Vector = [0] * l.shape[0]
    x: Vector = [0] * u.shape[0]

    for i in range(l.shape[0]):
        val_y = b[i]
        for k in range(i):
            val_y -= elem_from_csc(l, i, k) * y[k]
        y[i] = val_y

    for i in range(u.shape[0] - 1, -1, -1):
        val_x = y[i]
        for k in range(i + 1, u.shape[0]):
            val_x -= elem_from_csc(u, i, k) * x[k]
        x[i] = val_x / elem_from_csc(u, i, i)

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
        det_u *= elem_from_csc(u, i, i)

    return det_u * det_l


def elem_from_csc(csc_format: CSCMatrix, row: int, col: int) -> float:
    if row > csc_format.shape[0] - 1 or col > csc_format.shape[1] - 1:
        raise ValueError("no elem")

    my_data = csc_format.data[csc_format.indptr[col] : csc_format.indptr[col + 1]]
    my_col = csc_format.indices[csc_format.indptr[col] : csc_format.indptr[col + 1]]

    data_index = -1
    for index in range(len(my_col)):
        if my_col[index] == row:
            data_index = index
            break

    if data_index == -1:
        return 0

    return my_data[data_index]
 