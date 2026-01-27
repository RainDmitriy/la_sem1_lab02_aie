from CSC import CSCMatrix
from CSR import CSRMatrix
from type import Vector
from typing import Tuple, Optional, List


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    Ожидается, что матрица L хранит единицы на главной диагонали.
    """
    n = A.shape[0]
    if n != A.shape[1]:
        raise ValueError("Матрица должна быть квадратной для LU-разложения")
    
    # Преобразуем в плотный формат
    dense_A = A.to_dense()
    
    # Создаем копии для L и U
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    
    # LU-разложение без выбора ведущего элемента
    for i in range(n):
        # Верхняя треугольная матрица U
        for k in range(i, n):
            sum_val = 0.0
            for j in range(i):
                sum_val += L[i][j] * U[j][k]
            U[i][k] = dense_A[i][k] - sum_val
        
        # Проверка на вырожденность
        if abs(U[i][i]) < 1e-12:
            return None  # Матрица вырождена
        
        # Нижняя треугольная матрица L (с единицами на диагонали)
        L[i][i] = 1.0
        for k in range(i + 1, n):
            sum_val = 0.0
            for j in range(i):
                sum_val += L[k][j] * U[j][i]
            L[k][i] = (dense_A[k][i] - sum_val) / U[i][i]
    
    # Преобразуем обратно в CSC
    L_csc = CSCMatrix.from_dense(L)
    U_csc = CSCMatrix.from_dense(U)
    
    return L_csc, U_csc


def forward_substitution(L: CSCMatrix, b: Vector) -> Vector:
    """
    Прямой ход: решение Ly = b.
    L - нижняя треугольная матрица с единицами на диагонали.
    """
    n = L.shape[0]
    if n != len(b):
        raise ValueError("Размеры матрицы L и вектора b не совпадают")
    
    # Преобразуем L в плотный формат для упрощения
    L_dense = L.to_dense()
    
    y = [0.0] * n
    for i in range(n):
        sum_val = 0.0
        for j in range(i):
            sum_val += L_dense[i][j] * y[j]
        y[i] = b[i] - sum_val  # L[i][i] = 1, поэтому не делим
    
    return y


def backward_substitution(U: CSCMatrix, y: Vector) -> Vector:
    """
    Обратный ход: решение Ux = y.
    U - верхняя треугольная матрица.
    """
    n = U.shape[0]
    if n != len(y):
        raise ValueError("Размеры матрицы U и вектора y не совпадают")
    
    # Преобразуем U в плотный формат для упрощения
    U_dense = U.to_dense()
    
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_val = 0.0
        for j in range(i + 1, n):
            sum_val += U_dense[i][j] * x[j]
        
        # Проверка на ноль на диагонали
        if abs(U_dense[i][i]) < 1e-12:
            raise ValueError("Матрица U вырождена")
        
        x[i] = (y[i] - sum_val) / U_dense[i][i]
    
    return x


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    if A.shape[0] != len(b):
        raise ValueError("Размеры матрицы A и вектора b не совпадают")
    
    # Выполняем LU-разложение
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None  # LU-разложение не существует
    
    L, U = lu_result
    
    # Прямой ход: Ly = b
    y = forward_substitution(L, b)
    
    # Обратный ход: Ux = y
    x = backward_substitution(U, y)
    
    return x


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U) = 1 * произведение диагональных элементов U
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("Определитель существует только для квадратных матриц")
    
    # Выполняем LU-разложение
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return 0.0  # Если LU-разложение не существует, определитель = 0
    
    _, U = lu_result
    
    # Вычисляем определитель как произведение диагональных элементов U
    det = 1.0
    n = U.shape[0]
    
    # Преобразуем U в плотный формат
    U_dense = U.to_dense()
    
    for i in range(n):
        det *= U_dense[i][i]
    
    return det


def gaussian_elimination(A: CSCMatrix, b: Vector) -> Vector:
    """
    Решение СЛАУ методом Гаусса (для сравнения с LU-разложением).
    """
    n = A.shape[0]
    if n != len(b):
        raise ValueError("Размеры матрицы A и вектора b не совпадают")
    
    # Создаем расширенную матрицу
    dense_A = A.to_dense()
    aug_matrix = [row[:] + [b[i]] for i, row in enumerate(dense_A)]
    
    # Прямой ход метода Гаусса
    for i in range(n):
        # Поиск максимального элемента в столбце для устойчивости
        max_row = i
        max_val = abs(aug_matrix[i][i])
        for k in range(i + 1, n):
            if abs(aug_matrix[k][i]) > max_val:
                max_val = abs(aug_matrix[k][i])
                max_row = k
        
        # Меняем строки местами
        if max_row != i:
            aug_matrix[i], aug_matrix[max_row] = aug_matrix[max_row], aug_matrix[i]
        
        # Проверка на вырожденность
        if abs(aug_matrix[i][i]) < 1e-12:
            raise ValueError("Матрица вырождена")
        
        # Нормализация текущей строки
        pivot = aug_matrix[i][i]
        for j in range(i, n + 1):
            aug_matrix[i][j] /= pivot
        
        # Исключение переменной из последующих строк
        for k in range(i + 1, n):
            factor = aug_matrix[k][i]
            for j in range(i, n + 1):
                aug_matrix[k][j] -= factor * aug_matrix[i][j]
    
    # Обратный ход
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = aug_matrix[i][n]
        for j in range(i + 1, n):
            x[i] -= aug_matrix[i][j] * x[j]
    
    return x


def matrix_condition_number(A: CSCMatrix, norm_type: str = '1') -> float:
    """
    Вычисление числа обусловленности матрицы.
    norm_type: '1' для 1-нормы, 'inf' для бесконечной нормы
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("Число обусловленности определено только для квадратных матриц")
    
    dense_A = A.to_dense()
    n = A.shape[0]
    
    if norm_type == '1':
        # 1-норма: максимальная сумма по столбцам
        max_sum = 0.0
        for j in range(n):
            col_sum = 0.0
            for i in range(n):
                col_sum += abs(dense_A[i][j])
            max_sum = max(max_sum, col_sum)
        return max_sum
    
    elif norm_type == 'inf':
        # Бесконечная норма: максимальная сумма по строкам
        max_sum = 0.0
        for i in range(n):
            row_sum = 0.0
            for j in range(n):
                row_sum += abs(dense_A[i][j])
            max_sum = max(max_sum, row_sum)
        return max_sum
    
    else:
        raise ValueError("Неизвестный тип нормы. Используйте '1' или 'inf'")
