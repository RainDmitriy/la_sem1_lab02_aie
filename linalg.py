from CSC import CSCMatrix
from type import Vector
from typing import Tuple, Optional

EPS = 1e-10

def lu_decomposition(mat: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    Выполняет LU‑разложение матрицы в формате CSC.
    Возвращает пару (L, U), где L – нижняя треугольная с единицами на диагонали,
    U – верхняя треугольная матрица.
    """
    size = mat.dims[0]
    
    L_by_col = [{} for _ in range(size)]
    U_by_row = [{} for _ in range(size)]
    
    # преобразуем входную матрицу в словарь строк для быстрого доступа
    rows_dict = [{} for _ in range(size)]
    for col in range(size):
        for idx in range(mat.col_ptr[col], mat.col_ptr[col + 1]):
            row = mat.rows[idx]
            rows_dict[row][col] = float(mat.vals[idx])
    
    # массив для накопления обновлений
    updates = [{} for _ in range(size)]
    
    for step in range(size):
        current_row = {}
        
        # собираем элементы из исходной матрицы
        for j, val in rows_dict[step].items():
            if j >= step:
                current_row[j] = float(val)
        
        # добавляем накопленные обновления
        for j, val in updates[step].items():
            if j >= step:
                current_row[j] = current_row.get(j, 0.0) + float(val)
        
        diag_elem = current_row.get(step, 0.0)
        if abs(diag_elem) < EPS:
            return None
        
        U_by_row[step] = {}
        for j, val in current_row.items():
            if j >= step and abs(val) > EPS:
                U_by_row[step][j] = val
        
        U_by_row[step][step] = diag_elem
        L_by_col[step][step] = 1.0
        
        for i in range(step + 1, size):
            elem = 0.0
            if step in rows_dict[i]:
                elem += float(rows_dict[i][step])
            if step in updates[i]:
                elem += float(updates[i][step])
            
            if abs(elem) > EPS:
                factor = elem / diag_elem
                L_by_col[step][i] = factor
                
                for j, u_val in U_by_row[step].items():
                    if j > step:
                        delta = -factor * u_val
                        if abs(delta) > EPS:
                            updates[i][j] = updates[i].get(j, 0.0) + delta
    
    # преобразуем L в CSC
    L_vals, L_rows, L_colptr = [], [], [0]
    for col in range(size):
        sorted_rows = sorted(L_by_col[col].keys())
        for row in sorted_rows:
            L_vals.append(L_by_col[col][row])
            L_rows.append(row)
        L_colptr.append(len(L_vals))
    
    # преобразуем U в CSC (транспонируем словарь строк)
    U_by_col = [{} for _ in range(size)]
    for i in range(size):
        for j, val in U_by_row[i].items():
            U_by_col[j][i] = val
    
    U_vals, U_rows, U_colptr = [], [], [0]
    for col in range(size):
        sorted_rows = sorted(U_by_col[col].keys())
        for row in sorted_rows:
            U_vals.append(U_by_col[col][row])
            U_rows.append(row)
        U_colptr.append(len(U_vals))
    
    return (CSCMatrix(L_vals, L_rows, L_colptr, (size, size)),
            CSCMatrix(U_vals, U_rows, U_colptr, (size, size)))


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решает систему линейных уравнений Ax = b с использованием LU‑разложения.
    """
    lu_pair = lu_decomposition(A)
    if lu_pair is None:
        return None
    
    L, U = lu_pair
    n = A.dims[0]
    
    L_dense = L.to_dense()
    U_dense = U.to_dense()
    
    # прямой ход: Ly = b
    y = [0.0] * n
    for i in range(n):
        total = 0.0
        for j in range(i):
            total += L_dense[i][j] * y[j]
        y[i] = b[i] - total
    
    # обратный ход: Ux = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        total = 0.0
        for j in range(i + 1, n):
            total += U_dense[i][j] * x[j]
        
        if abs(U_dense[i][i]) < EPS:
            return None
        
        x[i] = (y[i] - total) / U_dense[i][i]
    
    return x

def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Вычисляет определитель матрицы с помощью LU‑разложения.
    """
    lu_pair = lu_decomposition(A)
    if lu_pair is None:
        return None
    
    _, U = lu_pair
    n = A.dims[0]
    
    U_dense = U.to_dense()
    det_value = 1.0
    for i in range(n):
        det_value *= U_dense[i][i]
    
    return det_value