from CSC import CSCMatrix
from CSR import CSRMatrix
from COO import COOMatrix
from type import Vector
from typing import Tuple, Optional, List, Dict
import math


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix, List[int]]]:
    """
    LU-разложение для CSC матрицы с частичным выбором ведущего элемента.
    Работает полностью с разреженными структурами, без конвертации в dense.
    Возвращает (L, U, P) где P - вектор перестановок.
    """
    n = A.shape[0]
    
    if n != A.shape[1]:
        raise ValueError("LU-разложение требует квадратную матрицу")
    
    # Конвертируем в CSR для удобства доступа по строкам
    A_csr = A._to_csr()
    
    # Инициализируем структуры данных
    # Храним L и U в формате словарей строк: row_dict[i][j] = значение
    L_rows = [{} for _ in range(n)]  # L - нижняя треугольная (i >= j)
    U_rows = [{} for _ in range(n)]  # U - верхняя треугольная (i <= j)
    
    # Вектор перестановок (изначально тождественная перестановка)
    P = list(range(n))
    
    # Копируем исходную матрицу в U (будем модифицировать в процессе)
    for i in range(n):
        row_dict = A_csr.get_row_as_dict(i)
        # Фильтруем только элементы с j >= 0 (все)
        for j, val in row_dict.items():
            if abs(val) > 1e-14:
                U_rows[i][j] = val
    
    # Основной цикл LU-разложения с частичным выбором
    for k in range(n):
        # ЧАСТИЧНЫЙ ВЫБОР (partial pivoting)
        # Ищем максимальный элемент в столбце k, начиная с строки k
        max_val = 0.0
        max_row = k
        
        for i in range(k, n):
            if k in U_rows[i]:
                val = abs(U_rows[i][k])
                if val > max_val:
                    max_val = val
                    max_row = i
        
        # Проверка на вырожденность
        if max_val < 1e-12:
            # Пробуем найти ненулевой элемент в других столбцах
            for j in range(k + 1, n):
                for i in range(k, n):
                    if j in U_rows[i]:
                        val = abs(U_rows[i][j])
                        if val > max_val:
                            max_val = val
                            max_row = i
                            # Меняем столбцы местами (это сложнее)
            
            if max_val < 1e-12:
                return None  # Матрица вырожденная
        
        # Перестановка строк, если необходимо
        if max_row != k:
            # Меняем местами строки k и max_row
            P[k], P[max_row] = P[max_row], P[k]
            U_rows[k], U_rows[max_row] = U_rows[max_row], U_rows[k]
            L_rows[k], L_rows[max_row] = L_rows[max_row], L_rows[k]
            
            # Также нужно обновить уже вычисленные L элементы для строк < k
            for i in range(k):
                if k in L_rows[i] and max_row in L_rows[i]:
                    L_rows[i][k], L_rows[i][max_row] = L_rows[i][max_row], L_rows[i][k]
        
        # Диагональный элемент U[k][k]
        u_kk = U_rows[k].get(k, 0.0)
        if abs(u_kk) < 1e-14:
            # Если после перестановки диагональ все еще 0
            return None
        
        # Записываем диагональный элемент L
        L_rows[k][k] = 1.0
        
        # Обрабатываем строки ниже k
        rows_to_update = []
        for i in range(k + 1, n):
            if k in U_rows[i]:
                rows_to_update.append(i)
        
        # Оптимизация: вычисляем все множители заранее
        multipliers = {}
        for i in rows_to_update:
            multipliers[i] = U_rows[i][k] / u_kk
            L_rows[i][k] = multipliers[i]
        
        # Обновляем строки U с учетом fill-in
        # ВАЖНО: учитываем, что при вычитании могут появиться новые ненулевые элементы
        for i in rows_to_update:
            multiplier = multipliers[i]
            
            # Создаем копию строки k U для безопасной итерации
            uk_items = list(U_rows[k].items())
            
            for j, u_kj in uk_items:
                if j >= k:  # Только элементы справа от диагонали
                    # Вычисляем новое значение
                    current = U_rows[i].get(j, 0.0)
                    new_val = current - multiplier * u_kj
                    
                    # Обновляем или удаляем элемент
                    if abs(new_val) > 1e-12:
                        U_rows[i][j] = new_val
                    elif j in U_rows[i]:
                        del U_rows[i][j]
            
            # Удаляем элемент (i, k) из U (он перешел в L)
            if k in U_rows[i]:
                del U_rows[i][k]
        
        # Очищаем столбец k в U (кроме диагонального)
        for i in range(k + 1, n):
            if k in U_rows[i] and i != k:
                del U_rows[i][k]
    
    # Очищаем очень маленькие значения для экономии памяти
    for i in range(n):
        # Очищаем L
        to_remove = []
        for j in L_rows[i]:
            if abs(L_rows[i][j]) < 1e-12:
                to_remove.append(j)
        for j in to_remove:
            del L_rows[i][j]
        
        # Очищаем U
        to_remove = []
        for j in U_rows[i]:
            if abs(U_rows[i][j]) < 1e-12:
                to_remove.append(j)
        for j in to_remove:
            del U_rows[i][j]
    
    # Конвертируем L и U в CSC формат
    L_csc = _sparse_rows_to_csc(L_rows, n, is_lower=True)
    U_csc = _sparse_rows_to_csc(U_rows, n, is_lower=False)
    
    return L_csc, U_csc, P


def _sparse_rows_to_csc(rows: List[Dict[int, float]], n: int, is_lower: bool) -> CSCMatrix:
    """
    Конвертирует список словарей строк в CSCMatrix.
    is_lower=True для нижней треугольной матрицы, False для верхней.
    """
    # Сначала собираем все ненулевые элементы
    elements = []
    
    for i in range(n):
        for j, val in rows[i].items():
            if abs(val) > 1e-14:
                if (is_lower and i >= j) or (not is_lower and i <= j):
                    elements.append((i, j, val))
    
    if not elements:
        return CSCMatrix([], [], [0] * (n + 1), (n, n))
    
    # Сортируем по столбцам, затем по строкам
    elements.sort(key=lambda x: (x[1], x[0]))
    
    # Строим CSC массивы
    data = []
    indices = []
    indptr = [0] * (n + 1)
    
    current_col = -1
    for i, j, val in elements:
        data.append(val)
        indices.append(i)
        
        # Обновляем indptr
        while current_col < j:
            current_col += 1
            indptr[current_col + 1] = indptr[current_col]
        
        indptr[j + 1] += 1
    
    # Заполняем оставшиеся столбцы
    for col in range(current_col + 1, n):
        indptr[col + 1] = indptr[col]
    
    return CSCMatrix(data, indices, indptr, (n, n))


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение с перестановками.
    Полностью работает с разреженными структурами.
    """
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None
    
    L, U, P = lu_result
    n = len(b)
    
    # Применяем перестановку к вектору b: b_perm = P * b
    b_perm = [b[P[i]] for i in range(n)]
    
    # Конвертируем L в CSR для удобного доступа по строкам
    L_csr = L._to_csr()
    
    # Решение Ly = b_perm (прямая подстановка)
    y = [0.0] * n
    
    for i in range(n):
        # Получаем ненулевые элементы строки i матрицы L
        row_dict = L_csr.get_row_as_dict(i)
        
        # Суммируем вклад уже вычисленных y[j]
        sum_val = 0.0
        for j, l_ij in row_dict.items():
            if j < i:  # Только элементы левее диагонали
                sum_val += l_ij * y[j]
        
        # Диагональный элемент L всегда 1.0
        y[i] = b_perm[i] - sum_val
    
    # Конвертируем U в CSR для удобного доступа по строкам
    U_csr = U._to_csr()
    
    # Решение Ux = y (обратная подстановка)
    x = [0.0] * n
    
    for i in range(n - 1, -1, -1):
        # Получаем ненулевые элементы строки i матрицы U
        row_dict = U_csr.get_row_as_dict(i)
        
        # Суммируем вклад уже вычисленных x[j]
        sum_val = 0.0
        diag_val = 0.0
        
        for j, u_ij in row_dict.items():
            if j > i:  # Элементы правее диагонали
                sum_val += u_ij * x[j]
            elif j == i:  # Диагональный элемент
                diag_val = u_ij
        
        if abs(diag_val) < 1e-14:
            return None
        
        x[i] = (y[i] - sum_val) / diag_val
    
    return x


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = (-1)^s * product(U[i][i]), где s - количество перестановок.
    """
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None
    
    _, U, P = lu_result
    n = A.shape[0]
    
    # Вычисляем знак перестановки
    # Количество транспозиций = четность перестановки
    visited = [False] * n
    transpositions = 0
    
    for i in range(n):
        if not visited[i]:
            j = i
            cycle_len = 0
            while not visited[j]:
                visited[j] = True
                j = P[j]
                cycle_len += 1
            if cycle_len > 1:
                transpositions += cycle_len - 1
    
    sign = 1 if transpositions % 2 == 0 else -1
    
    # Произведение диагональных элементов U
    det = sign
    U_csr = U._to_csr()
    
    for i in range(n):
        diag_val = U_csr.get(i, i)
        if abs(diag_val) < 1e-14:
            return 0.0
        det *= diag_val
    
    return det

