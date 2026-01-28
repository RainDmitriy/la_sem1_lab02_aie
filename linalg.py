from CSC import CSCMatrix
from typing import Tuple, Optional, List
Vector = List[float]


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix, List[int]]]:
    """
    LU-разложение для CSC матрицы с выбором главного элемента.
    Возвращает (L, U, P) - нижнюю и верхнюю треугольные матрицы и вектор перестановок.
    """
    n = A.shape[0]
    assert A.shape[0] == A.shape[1], "Matrix must be square for LU decomposition"

    dense_A = A.to_dense()
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    P = list(range(n))

    # Создаем копию матрицы для модификаций
    A_temp = [row[:] for row in dense_A]

    for i in range(n):
        # Выбор главного элемента (partial pivoting)
        max_row = i
        max_val = abs(A_temp[i][i])
        
        for k in range(i + 1, n):
            if abs(A_temp[k][i]) > max_val:
                max_val = abs(A_temp[k][i])
                max_row = k
        
        if max_val < 1e-15:
            return None  # Матрица вырождена
        
        # Перестановка строк
        if max_row != i:
            A_temp[i], A_temp[max_row] = A_temp[max_row], A_temp[i]
            P[i], P[max_row] = P[max_row], P[i]
            # Также переставляем уже вычисленные элементы L
            for j in range(i):
                L[i][j], L[max_row][j] = L[max_row][j], L[i][j]
        
        # Вычисление элементов U и L
        for j in range(i, n):
            # Вычисляем U[i][j]
            s = 0.0
            for k in range(i):
                s += L[i][k] * U[k][j]
            U[i][j] = A_temp[i][j] - s
        
        # Проверка на нулевой диагональный элемент
        if abs(U[i][i]) < 1e-15:
            return None
        
        for j in range(i, n):
            if i == j:
                L[j][i] = 1.0
            else:
                # Вычисляем L[j][i]
                s = 0.0
                for k in range(i):
                    s += L[j][k] * U[k][i]
                L[j][i] = (A_temp[j][i] - s) / U[i][i]

    # Преобразуем L и U в разреженный формат
    from COO import COOMatrix
    
    L_data, L_rows, L_cols = [], [], []
    U_data, U_rows, U_cols = [], [], []
    
    for i in range(n):
        for j in range(n):
            if j <= i and abs(L[i][j]) > 1e-15:
                L_data.append(L[i][j])
                L_rows.append(i)
                L_cols.append(j)
            
            if j >= i and abs(U[i][j]) > 1e-15:
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
    
    # Используем плотное представление для простоты
    dense_L = L.to_dense()
    
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += dense_L[i][j] * y[j]
        y[i] = b[i] - s
    
    return y


def solve_upper_triangular(U: CSCMatrix, y: Vector) -> Vector:
    """Решение Ux = y (U - верхняя треугольная)."""
    n = U.shape[0]
    x = [0.0] * n
    
    # Используем плотное представление для простоты
    dense_U = U.to_dense()
    
    for i in range(n - 1, -1, -1):
        if abs(dense_U[i][i]) < 1e-15:
            raise ValueError("Matrix U is singular")
        
        s = 0.0
        for j in range(i + 1, n):
            s += dense_U[i][j] * x[j]
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
    
    # Применяем перестановку к вектору b: b' = Pb
    b_permuted = [b[P[i]] for i in range(len(b))]
    
    y = solve_lower_triangular(L, b_permuted)
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
    
    # Определитель U
    det_U = 1.0
    dense_U = U.to_dense()
    for i in range(n):
        det_U *= dense_U[i][i]
    
    # Учет перестановок (четность перестановок)
    swaps = 0
    for i in range(n):
        if P[i] != i:
            swaps += 1
    
    # Если swaps нечетное, меняем знак
    if swaps % 2 == 1:
        det_U = -det_U
    
    return det_U
