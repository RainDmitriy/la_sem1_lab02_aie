from CSC import CSCMatrix
from CSR import CSRMatrix
from type import Vector
from typing import Tuple, Optional


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """LU-разложение (A = L * U). L имеет 1 на диагонали."""
    n, m = A.shape
    if n != m:
        return None

    mat = A.to_dense()
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        L[i][i] = 1.0  # По определению задания

        # Считаем строку U
        for j in range(i, n):
            sum_lu = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = mat[i][j] - sum_lu

        # Считаем столбец L
        for j in range(i + 1, n):
            if abs(U[i][i]) < 1e-15:  # Проверка на нулевой минор
                return None
            sum_lu = sum(L[j][k] * U[k][i] for k in range(i))
            L[j][i] = (mat[j][i] - sum_lu) / U[i][i]

    return CSCMatrix.from_dense(L), CSCMatrix.from_dense(U)


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """Решение СЛАУ через два этапа: Ly = b и Ux = y."""
    decomp = lu_decomposition(A)
    if not decomp:
        return None

    L_mat, U_mat = decomp
    L, U = L_mat.to_dense(), U_mat.to_dense()
    n = len(b)

    # Прямой ход (L * y = b)
    y = [0.0] * n
    for i in range(n):
        s = sum(L[i][j] * y[j] for j in range(i))
        y[i] = b[i] - s

    # Обратный ход (U * x = y)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(U[i][i]) < 1e-15:
            return None
        s = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - s) / U[i][i]

    return x


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """Определитель — это произведение диагонали матрицы U."""
    decomp = lu_decomposition(A)
    if not decomp:
        return 0.0

    _, U_sparse = decomp
    U = U_sparse.to_dense()

    determinant = 1.0
    for i in range(A.shape[0]):
        determinant *= U[i][i]

    return determinant