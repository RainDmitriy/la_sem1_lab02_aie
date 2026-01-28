from CSC import CSCMatrix
from CSR import CSRMatrix
from COO import COOMatrix
from type import Vector
from typing import Tuple, Optional, List, Dict
import math


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix, List[int]]]:
    """
    LU-разложение для CSC матрицы с частичным выбором ведущего элемента.
    Работает полностью с разреженными структурами.
    """
    n = A.shape[0]
    
    if n != A.shape[1]:
        raise ValueError("LU-разложение требует квадратную матрицу")
    
    # Конвертируем в CSR для удобства доступа по строкам
    A_csr = A._to_csr()
    
    # Инициализируем L и U как списки словарей
    L_rows = [{} for _ in range(n)]  # L[i][j] для i >= j
    U_rows = [{} for _ in range(n)]  # U[i][j] для i <= j
    
    # Копируем A в U
    for i in range(n):
        row_dict = A_csr.get_row_as_dict(i)
        for j, val in row_dict.items():
            if abs(val) > 1e-14:
                U_rows[i][j] = val
    
    # Вектор перестановок
    P = list(range(n))
    
    # Основной цикл LU-разложения
    for k in range(n):
        # Частичный выбор ведущего элемента в столбце k
        max_val = 0.0
        max_row = k
        
        # Ищем максимальный элемент в столбце k (от строки k до n-1)
        for i in range(k, n):
            if k in U_rows[i]:
                val = abs(U_rows[i][k])
                if val > max_val:
                    max_val = val
                    max_row = i
        
        if max_val < 1e-12:
            # Если не нашли ненулевой элемент, ищем в других столбцах
            for j in range(k + 1, n):
                for i in range(k, n):
                    if j in U_rows[i]:
                        val = abs(U_rows[i][j])
                        if val > max_val:
                            max_val = val
                            max_row = i
            
            if max_val < 1e-12:
                return None  # Матрица вырожденная
        
        # Перестановка строк, если необходимо
        if max_row != k:
            # Меняем строки в U
            U_rows[k], U_rows[max_row] = U_rows[max_row], U_rows[k]
            
            # Меняем строки в L (только уже вычисленные элементы)
            for j in range(k):
                if k in L_rows[max_row] and j in L_rows[k]:
                    L_rows[k][j], L_rows[max_row][j] = L_rows[max_row][j], L_rows[k][j]
                elif k in L_rows[max_row]:
                    L_rows[k][j] = L_rows[max_row][j]
                    del L_rows[max_row][j]
                elif j in L_rows[k]:
                    L_rows[max_row][j] = L_rows[k][j]
                    del L_rows[k][j]
            
            # Обновляем перестановку
            P[k], P[max_row] = P[max_row], P[k]
        
        # Диагональный элемент U
        u_kk = U_rows[k].get(k, 0.0)
        if abs(u_kk) < 1e-14:
            return None
        
        # Устанавливаем диагональный элемент L = 1
        L_rows[k][k] = 1.0
        
        # Обновляем строки ниже k
        for i in range(k + 1, n):
            if k in U_rows[i]:
                # Вычисляем множитель
                l_ik = U_rows[i][k] / u_kk
                L_rows[i][k] = l_ik
                
                # Обновляем строку i матрицы U
                # U[i][j] = U[i][j] - l_ik * U[k][j] для j >= k
                for j, u_kj in list(U_rows[k].items()):
                    if j >= k:
                        new_val = U_rows[i].get(j, 0.0) - l_ik * u_kj
                        if abs(new_val) > 1e-12:
                            U_rows[i][j] = new_val
                        elif j in U_rows[i]:
                            del U_rows[i][j]
                
                # Удаляем элемент (i, k) из U (он теперь в L)
                if k in U_rows[i]:
                    del U_rows[i][k]
    
    # Очищаем очень маленькие значения
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
    L_csc = _sparse_dict_to_csc(L_rows, n, is_lower=True)
    U_csc = _sparse_dict_to_csc(U_rows, n, is_lower=False)
    
    return L_csc, U_csc, P


def _sparse_dict_to_csc(rows: List[Dict[int, float]], n: int, is_lower: bool) -> CSCMatrix:
    """Конвертирует список словарей строк в CSCMatrix."""
    # Собираем все элементы
    elements = []
    
    for i in range(n):
        for j, val in rows[i].items():
            if abs(val) > 1e-14:
                # Проверяем треугольность
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
        
        # Заполняем indptr пропущенных столбцов
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
    Решение СЛАУ Ax = b через LU-разложение.
    """
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None
    
    L, U, P = lu_result
    n = len(b)
    
    # Применяем перестановку к b
    b_perm = [b[P[i]] for i in range(n)]
    
    # Конвертируем L в CSR для удобного доступа
    L_csr = L._to_csr()
    
    # Решение Ly = b_perm (прямая подстановка)
    y = [0.0] * n
    for i in range(n):
        sum_val = 0.0
        row_dict = L_csr.get_row_as_dict(i)
        
        for j, val in row_dict.items():
            if j < i:  # Нижняя треугольная часть
                sum_val += val * y[j]
        
        # Диагональный элемент L всегда 1.0
        y[i] = b_perm[i] - sum_val
    
    # Конвертируем U в CSR для удобного доступа
    U_csr = U._to_csr()
    
    # Решение Ux = y (обратная подстановка)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_val = 0.0
        row_dict = U_csr.get_row_as_dict(i)
        
        diag_val = 0.0
        for j, val in row_dict.items():
            if j > i:  # Верхняя треугольная часть
                sum_val += val * x[j]
            elif j == i:  # Диагональный элемент
                diag_val = val
        
        if abs(diag_val) < 1e-14:
            return None
        
        x[i] = (y[i] - sum_val) / diag_val
    
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
    
    # Вычисляем знак перестановки
    # Подсчитываем количество транспозиций
    visited = [False] * n
    swaps = 0
    
    for i in range(n):
        if not visited[i]:
            cycle_len = 0
            j = i
            while not visited[j]:
                visited[j] = True
                j = P[j]
                cycle_len += 1
            if cycle_len > 1:
                swaps += cycle_len - 1
    
    sign = 1 if swaps % 2 == 0 else -1
    
    # Произведение диагональных элементов U
    det = sign
    for i in range(n):
        diag = U.get(i, i)
        if abs(diag) < 1e-14:
            return 0.0
        det *= diag
    
    return det