from CSC import CSCMatrix
from CSR import CSRMatrix
from types import Vector
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
            s = 0.0
            for j in range(i):
                s += L[i][j] * U[j][k]
            U[i][k] = dense_A[i][k] - s
        
        # Проверка на вырожденность
        if abs(U[i][i]) < 1e-12:
            return None
        
        # Нижняя треугольная матрица L
        for k in range(i, n):
            if i == k:
                L[i][i] = 1.0
            else:
                s = 0.0
                for j in range(i):
                    s += L[k][j] * U[j][i]
                L[k][i] = (dense_A[k][i] - s) / U[i][i]
    
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
    
    L_dense = L.to_dense()
    y = [0.0] * n
    
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L_dense[i][j] * y[j]
        y[i] = b[i] - s  # L[i][i] = 1
    
    return y


def backward_substitution(U: CSCMatrix, y: Vector) -> Vector:
    """
    Обратный ход: решение Ux = y.
    U - верхняя треугольная матрица.
    """
    n = U.shape[0]
    if n != len(y):
        raise ValueError("Размеры матрицы U и вектора y не совпадают")
    
    U_dense = U.to_dense()
    x = [0.0] * n
    
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += U_dense[i][j] * x[j]
        
        if abs(U_dense[i][i]) < 1e-12:
            raise ValueError("Матрица вырождена")
        
        x[i] = (y[i] - s) / U_dense[i][i]
    
    return x


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    if A.shape[0] != len(b):
        raise ValueError("Размеры матрицы A и вектора b не совпадают")
    
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None
    
    L, U = lu_result
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    
    return x


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U) = 1 * произведение диагональных элементов U
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("Определитель существует только для квадратных матриц")
    
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return 0.0
    
    _, U = lu_result
    U_dense = U.to_dense()
    n = A.shape[0]
    
    det = 1.0
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
    
    dense_A = A.to_dense()
    
    # Копируем матрицу и вектор
    aug = [row[:] for row in dense_A]
    for i in range(n):
        aug[i].append(b[i])
    
    # Прямой ход
    for i in range(n):
        # Поиск главного элемента
        max_row = i
        max_val = abs(aug[i][i])
        for k in range(i + 1, n):
            if abs(aug[k][i]) > max_val:
                max_val = abs(aug[k][i])
                max_row = k
        
        # Перестановка строк
        if max_row != i:
            aug[i], aug[max_row] = aug[max_row], aug[i]
        
        # Проверка на вырожденность
        if abs(aug[i][i]) < 1e-12:
            raise ValueError("Матрица вырождена")
        
        # Нормализация
        pivot = aug[i][i]
        for j in range(i, n + 1):
            aug[i][j] /= pivot
        
        # Исключение
        for k in range(i + 1, n):
            factor = aug[k][i]
            for j in range(i, n + 1):
                aug[k][j] -= factor * aug[i][j]
    
    # Обратный ход
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = aug[i][n]
        for j in range(i + 1, n):
            x[i] -= aug[i][j] * x[j]
    
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
        norm = 0.0
        for j in range(n):
            col_sum = sum(abs(dense_A[i][j]) for i in range(n))
            norm = max(norm, col_sum)
        return norm
    
    elif norm_type == 'inf':
        # inf-норма: максимальная сумма по строкам
        norm = 0.0
        for i in range(n):
            row_sum = sum(abs(dense_A[i][j]) for j in range(n))
            norm = max(norm, row_sum)
        return norm
    
    else:
        raise ValueError("Неизвестный тип нормы")
