from CSC import CSCMatrix
from CSR import CSRMatrix
from type import Vector
from typing import Tuple, Optional, List, Dict


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы
    Ожидается, что матрица L хранит единицы на главной диагонали
    """
    num_rows, num_cols = A.shape
    if num_rows != num_cols:
        return None

    adjacency_by_row: List[Dict[int, float]] = [
        {} for _ in range(num_rows)
    ]
    adjacency_by_column: List[Dict[int, float]] = [
        {} for _ in range(num_cols)
    ]
    
    # Преобразование в удобный формат
    for current_col in range(num_cols):
        start_ptr = A.indptr[current_col]
        end_ptr = A.indptr[current_col + 1]
        
        for position in range(start_ptr, end_ptr):
            current_row = A.indices[position]
            element_value = A.data[position]
            
            adjacency_by_row[current_row][current_col] = element_value
            adjacency_by_column[current_col][current_row] = element_value
    
    # Подготовка данных для L и U
    lower_matrix_data: List[Tuple[int, int, float]] = []
    upper_matrix_data: List[Tuple[int, int, float]] = []
    
    # Основной процесс разложения
    for diagonal_position in range(num_rows):
        # Получение диагонального элемента
        diag_element = adjacency_by_column[diagonal_position].get(
            diagonal_position, 0.0
        )
        
        if abs(diag_element) < 1e-15:
            return None
        
        # Добавление диагональных элементов
        upper_matrix_data.append(
            (diagonal_position, diagonal_position, diag_element)
        )
        lower_matrix_data.append(
            (diagonal_position, diagonal_position, 1.0)
        )
        
        # Обработка строки для U
        upper_columns: List[int] = []
        upper_values: List[float] = []
        
        if adjacency_by_row[diagonal_position]:
            sorted_row_items = sorted(
                adjacency_by_row[diagonal_position].items()
            )
            
            for col, val in sorted_row_items:
                if col <= diagonal_position:
                    continue
                    
                if abs(val) > 1e-15:
                    upper_matrix_data.append(
                        (diagonal_position, col, val)
                    )
                    upper_columns.append(col)
                    upper_values.append(val)
        
        # Обработка столбца для L
        lower_rows: List[int] = []
        lower_values: List[float] = []
        
        if adjacency_by_column[diagonal_position]:
            sorted_col_items = sorted(
                adjacency_by_column[diagonal_position].items()
            )
            
            for row, val in sorted_col_items:
                if row <= diagonal_position:
                    continue
                    
                calculated_value = val / diag_element
                
                if abs(calculated_value) > 1e-15:
                    lower_matrix_data.append(
                        (row, diagonal_position, calculated_value)
                    )
                    lower_rows.append(row)
                    lower_values.append(calculated_value)
        
        # Обновление оставшейся части матрицы
        for l_index in range(len(lower_rows)):
            current_l_row = lower_rows[l_index]
            current_l_value = lower_values[l_index]
            
            for u_index in range(len(upper_columns)):
                current_u_col = upper_columns[u_index]
                current_u_value = upper_values[u_index]
                
                update_value = current_l_value * current_u_value
                
                # Обновление значений
                old_row_val = adjacency_by_row[current_l_row].get(
                    current_u_col, 0.0
                )
                new_row_val = old_row_val - update_value
                
                if abs(new_row_val) > 1e-15:
                    adjacency_by_row[current_l_row][current_u_col] = new_row_val
                    adjacency_by_column[current_u_col][current_l_row] = new_row_val
                else:
                    # Удаление нулевых элементов
                    if current_u_col in adjacency_by_row[current_l_row]:
                        del adjacency_by_row[current_l_row][current_u_col]
                    if current_l_row in adjacency_by_column[current_u_col]:
                        del adjacency_by_column[current_u_col][current_l_row]
    
    # Функция преобразования триплетов в CSC формат
    def create_csc_from_triplets(
        triplets: List[Tuple[int, int, float]], 
        matrix_size: int
    ) -> CSCMatrix:
        # Сортировка по столбцам, затем по строкам
        sorted_triplets = sorted(
            triplets, 
            key=lambda item: (item[1], item[0])
        )
        
        # Извлечение данных
        values = [t[2] for t in sorted_triplets]
        row_indices = [t[0] for t in sorted_triplets]
        
        # Создание indptr
        column_pointers = [0] * (matrix_size + 1)
        current_col = 0
        
        for _, col, _ in sorted_triplets:
            while current_col < col:
                current_col += 1
                column_pointers[current_col + 1] = column_pointers[current_col]
            column_pointers[current_col + 1] += 1
        
        # Заполнение оставшихся столбцов
        while current_col < matrix_size - 1:
            current_col += 1
            column_pointers[current_col + 1] = column_pointers[current_col]
        
        return CSCMatrix(
            values, 
            row_indices, 
            column_pointers, 
            (matrix_size, matrix_size)
        )
    
    L_matrix = create_csc_from_triplets(lower_matrix_data, num_rows)
    U_matrix = create_csc_from_triplets(upper_matrix_data, num_rows)
    
    return (L_matrix, U_matrix)


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    decomposition_result = lu_decomposition(A)
    
    if decomposition_result is None:
        return None
    
    L, U = decomposition_result
    system_size = len(b)
    
    # Прямая подстановка для Ly = b
    intermediate_solution = [value for value in b]
    
    for col_idx in range(system_size):
        if abs(intermediate_solution[col_idx]) < 1e-15:
            continue
            
        col_start = L.indptr[col_idx]
        col_end = L.indptr[col_idx + 1]
        
        for pos in range(col_start, col_end):
            row_idx = L.indices[pos]
            element_val = L.data[pos]
            
            if row_idx > col_idx:
                intermediate_solution[row_idx] -= (
                    element_val * intermediate_solution[col_idx]
                )
    
    # Обратная подстановка для Ux = y
    final_solution = [val for val in intermediate_solution]
    
    for col_idx in reversed(range(system_size)):
        # Поиск диагонального элемента
        diag_found = False
        diag_val = 0.0
        
        col_start = U.indptr[col_idx]
        col_end = U.indptr[col_idx + 1]
        
        for pos in range(col_start, col_end):
            if U.indices[pos] == col_idx:
                diag_val = U.data[pos]
                diag_found = True
                break
        
        if not diag_found or abs(diag_val) < 1e-15:
            return None
        
        # Деление на диагональный элемент
        final_solution[col_idx] /= diag_val
        
        if abs(final_solution[col_idx]) < 1e-15:
            continue
        
        # Обновление остальных компонент
        for pos in range(col_start, col_end):
            row_idx = U.indices[pos]
            element_val = U.data[pos]
            
            if row_idx < col_idx:
                final_solution[row_idx] -= (
                    element_val * final_solution[col_idx]
                )
    
    return final_solution


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    decomposition_result = lu_decomposition(A)
    
    if decomposition_result is None:
        return 0.0
    
    _, upper_matrix = decomposition_result
    matrix_size = upper_matrix.shape[0]
    determinant = 1.0
    
    for col_idx in range(matrix_size):
        diagonal_element = 0.0
        element_found = False
        
        col_start = upper_matrix.indptr[col_idx]
        col_end = upper_matrix.indptr[col_idx + 1]
        
        for pos in range(col_start, col_end):
            if upper_matrix.indices[pos] == col_idx:
                diagonal_element = upper_matrix.data[pos]
                element_found = True
                break
        
        if not element_found:
            return 0.0
        
        determinant *= diagonal_element
    
    return determinant
