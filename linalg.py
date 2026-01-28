from CSC import CSCMatrix
from type import Vector
from typing import Tuple, Optional, List


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix, List[int]]]:
    """
    LU-разложение для CSC матрицы.
    Возвращает (L, U, P) - нижнюю и верхнюю треугольные матрицы и вектор перестановок.
    """
    n = A.shape[0]
    assert A.shape[0] == A.shape[1], "Matrix must be square for LU decomposition"

    # Преобразуем в плотную матрицу
    dense_A = A.to_dense()

    L = [[0.0 for _ in range(n)] for _ in range(n)]
    U = [[0.0 for _ in range(n)] for _ in range(n)]
    P = list(range(n))

    # Копируем A для модификаций
    A_temp = [row[:] for row in dense_A]

    for i in range(n):
        # Поиск pivot элемента
        max_row = i
        max_val = abs(A_temp[i][i])
        
        for k in range(i + 1, n):
            if abs(A_temp[k][i]) > max_val:
                max_val = abs(A_temp[k][i])
                max_row = k
        
        if max_val < 1e-12:
            # Матрица вырождена
            return None
        
        if max_row != i:
            # Перестановка строк в A_temp
            A_temp[i], A_temp[max_row] = A_temp[max_row], A_temp[i]
            # Сохраняем перестановку
            P[i], P[max_row] = P[max_row], P[i]
        
        # Вычисление элементов U и L
        for j in range(i, n):
            # Вычисляем U[i][j]
            s = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = A_temp[i][j] - s
        
        for j in range(i, n):
            # Вычисляем L[j][i]
            if abs(U[i][i]) < 1e-12:
                return None
            s = sum(L[j][k] * U[k][i] for k in range(i))
            L[j][i] = (A_temp[j][i] - s) / U[i][i]
        
        L[i][i] = 1.0

    # Преобразуем L и U в разреженный формат
    from COO import COOMatrix
    
    L_data, L_rows, L_cols = [], [], []
    U_data, U_rows, U_cols = [], [], []
    
    for i in range(n):
        for j in range(n):
            if j <= i and abs(L[i][j]) > 1e-12:
                L_data.append(L[i][j])
                L_rows.append(i)
                L_cols.append(j)
            
            if j >= i and abs(U[i][j]) > 1e-12:
                U_data.append(U[i][j])
                U_rows.append(i)
                U_cols.append(j)
    
    L_coo = COOMatrix(L_data, L_rows, L_cols, (n, n))
    U_coo = COOMatrix(U_data, U_rows, U_cols, (n, n))
    
    return L_coo._to_csc(), U_coo._to_csc(), P


def solve_lower_triangular(L: CSCMatrix, b: Vector) -> Vector:
    """Решение Ly = b (L - нижняя треугольная с единицами на диагонали)."""
    n = L.shape[0]
    y = [0.0] * n
    dense_L = L.to_dense()
    
    for i in range(n):
        s = sum(dense_L[i][j] * y[j] for j in range(i))
        y[i] = b[i] - s
    
    return y


def solve_upper_triangular(U: CSCMatrix, y: Vector) -> Vector:
    """Решение Ux = y (U - верхняя треугольная)."""
    n = U.shape[0]
    x = [0.0] * n
    dense_U = U.to_dense()
    
    for i in range(n - 1, -1, -1):
        if abs(dense_U[i][i]) < 1e-12:
            raise ValueError("Matrix is singular")
        
        s = sum(dense_U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - s) / dense_U[i][i]
    
    return x


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None

    L, U, P = lu_result

    # Применяем перестановку к вектору b
    b_permuted = [b[P[i]] for i in range(len(b))]
    
    # Решаем Ly = Pb
    y = solve_lower_triangular(L, b_permuted)
    
    # Решаем Ux = y
    x = solve_upper_triangular(U, y)
    
    return x


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    """
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None

    _, U, P = lu_result
    n = A.shape[0]
    
    # Определитель U - произведение диагональных элементов
    det_U = 1.0
    dense_U = U.to_dense()
    for i in range(n):
        det_U *= dense_U[i][i]
    
    # Учет перестановок
    swaps = 0
    for i in range(n):
        if P[i] != i:
            # Ищем, куда перешел элемент i
            for j in range(i + 1, n):
                if P[j] == i:
                    swaps += 1
                    break
    
    det = det_U * (-1) ** swaps
    return det
