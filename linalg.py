from CSC import CSCMatrix
from CSR import CSRMatrix
from type import Vector
from typing import Tuple, Optional


def lu_decomposition(matrix: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы
    Возвращает нижнюю и верхнюю треугольные матрицы(L,U)
    Ожидается, что матрица L хранит единицы на главной диагонали
    """
    rows_count, columns_count = matrix.shape
    if rows_count != columns_count:
        return None
        
    rows_dictionary = [dict() for _ in range(rows_count)]
    columns_dictionary = [dict() for _ in range(rows_count)]
    
    for column_index in range(columns_count):
        for position in range(matrix.indptr[column_index], matrix.indptr[column_index + 1]):
            row_index = matrix.indices[position]
            value = matrix.data[position]
            rows_dictionary[row_index][column_index] = value
            columns_dictionary[column_index][row_index] = value
            
    lower_triplets = []
    upper_triplets = []
    
    for diagonal_index in range(rows_count):
        pivot_value = columns_dictionary[diagonal_index].get(diagonal_index, 0.0)
        if pivot_value == 0:
            return None
            
        upper_triplets.append((diagonal_index, diagonal_index, pivot_value))
        lower_triplets.append((diagonal_index, diagonal_index, 1.0))
        
        upper_indices = []
        upper_values = []
        
        if rows_dictionary[diagonal_index]:
            for column, value in sorted(rows_dictionary[diagonal_index].items()):
                if column > diagonal_index:
                    if abs(value) > 1e-15:
                        upper_triplets.append((diagonal_index, column, value))
                        upper_indices.append(column)
                        upper_values.append(value)
                        
        lower_indices = []
        lower_values = []
        
        if columns_dictionary[diagonal_index]:
            for row, value in sorted(columns_dictionary[diagonal_index].items()):
                if row > diagonal_index:
                    lower_value = value / pivot_value
                    if abs(lower_value) > 1e-15:
                        lower_triplets.append((row, diagonal_index, lower_value))
                        lower_indices.append(row)
                        lower_values.append(lower_value)
                        
        for lower_position in range(len(lower_indices)):
            row = lower_indices[lower_position]
            lower_value = lower_values[lower_position]
            
            for upper_position in range(len(upper_indices)):
                column = upper_indices[upper_position]
                upper_value = upper_values[upper_position]
                
                update = lower_value * upper_value
                old_value = rows_dictionary[row].get(column, 0.0)
                new_value = old_value - update
                
                if abs(new_value) > 1e-15:
                    rows_dictionary[row][column] = new_value
                    columns_dictionary[column][row] = new_value
                elif column in rows_dictionary[row]:
                    del rows_dictionary[row][column]
                    del columns_dictionary[column][row]

    def convert_to_csc(triplets_list, matrix_size):
        triplets_list.sort(key=lambda element: (element[1], element[0]))
        
        values = [element[2] for element in triplets_list]
        indices = [element[0] for element in triplets_list]
        
        indptr = [0] * (matrix_size + 1)
        current_column = 0
        
        for _, column, _ in triplets_list:
            while current_column < column:
                current_column += 1
                indptr[current_column + 1] = indptr[current_column]
            indptr[current_column + 1] += 1
            
        while current_column < matrix_size - 1:
            current_column += 1
            indptr[current_column + 1] = indptr[current_column]
            
        return CSCMatrix(values, indices, indptr, (matrix_size, matrix_size))
        
    return convert_to_csc(lower_triplets, rows_count), convert_to_csc(upper_triplets, rows_count)


def solve_SLAE_lu(matrix: CSCMatrix, right_side: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение
    """
    decomposition = lu_decomposition(matrix)
    if decomposition is None:
        return None
        
    lower_matrix, upper_matrix = decomposition
    size = len(right_side)
    
    intermediate_y = list(right_side)
    
    # Прямая подстановка для Ly = b
    for column_index in range(size):
        if intermediate_y[column_index] != 0:
            for position in range(lower_matrix.indptr[column_index], lower_matrix.indptr[column_index + 1]):
                row_index = lower_matrix.indices[position]
                value = lower_matrix.data[position]
                if row_index > column_index:
                    intermediate_y[row_index] -= value * intermediate_y[column_index]
                    
    solution_x = list(intermediate_y)
    
    # Обратная подстановка для Ux = y
    for column_index in range(size - 1, -1, -1):
        diagonal_value = 0
        
        for position in range(upper_matrix.indptr[column_index], upper_matrix.indptr[column_index + 1]):
            if upper_matrix.indices[position] == column_index:
                diagonal_value = upper_matrix.data[position]
                break
                
        if diagonal_value == 0:
            return None
            
        solution_x[column_index] /= diagonal_value
        
        if solution_x[column_index] != 0:
            for position in range(upper_matrix.indptr[column_index], upper_matrix.indptr[column_index + 1]):
                row_index = upper_matrix.indices[position]
                value = upper_matrix.data[position]
                if row_index < column_index:
                    solution_x[row_index] -= value * solution_x[column_index]
                    
    return solution_x


def find_det_with_lu(matrix: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение
    Формула: det(A) = det(L) * det(U)
    """
    decomposition = lu_decomposition(matrix)
    if decomposition is None:
        return 0.0
        
    _, upper_matrix = decomposition
    
    determinant = 1
    matrix_size = upper_matrix.shape[0]
    
    for column_index in range(matrix_size):
        diagonal_found = False
        
        for position in range(upper_matrix.indptr[column_index], upper_matrix.indptr[column_index + 1]):
            if upper_matrix.indices[position] == column_index:
                determinant *= upper_matrix.data[position]
                diagonal_found = True
                break
                
        if not diagonal_found:
            return 0.0
            
    return determinant