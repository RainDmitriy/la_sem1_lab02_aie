from CSC import CSCMatrix
from CSR import CSRMatrix
from type import Vector
from typing import Tuple, Optional


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение для CSC матрицы
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы
    Ожидается, что матрица L хранит единицы на главной диагонали
    """
    n, m = A.shape
    if n != m:
        return None

    row_adjacent = [dict() for _ in range(n)]
    column_adjacent = [dict() for _ in range(n)]
    
    for column_index in range(m):
        for k in range(A.indptr[column_index], A.indptr[column_index + 1]):
             row_index = A.indices[k]
             value = A.data[k]
             row_adjacent[row_index][column_index] = value
             column_adjacent[column_index][row_index] = value
    
    L_triplets_collection = []
    U_triplets_collection = []
    
    for diagonal_index in range(n):
        pivot_element = column_adjacent[diagonal_index].get(diagonal_index, 0.0)
        if pivot_element == 0:
            return None
        
        U_triplets_collection.append((diagonal_index, diagonal_index, pivot_element))
        L_triplets_collection.append((diagonal_index, diagonal_index, 1.0))

        upper_indices_list = []
        upper_values_list = []

        if row_adjacent[diagonal_index]:
            for column, value in sorted(row_adjacent[diagonal_index].items()):
                if column > diagonal_index:
                    if abs(value) > 1e-15:
                        U_triplets_collection.append((diagonal_index, column, value))
                        upper_indices_list.append(column)
                        upper_values_list.append(value)

        lower_indices_list = []
        lower_values_list = []

        if column_adjacent[diagonal_index]:
            for row, value in sorted(column_adjacent[diagonal_index].items()):
                if row > diagonal_index:
                    lower_value = value / pivot_element
                    if abs(lower_value) > 1e-15:
                        L_triplets_collection.append((row, diagonal_index, lower_value))
                        lower_indices_list.append(row)
                        lower_values_list.append(lower_value)

        for idx in range(len(lower_indices_list)):
            current_row = lower_indices_list[idx]
            current_lower_value = lower_values_list[idx]

            for jdx in range(len(upper_indices_list)):
                current_column = upper_indices_list[jdx]
                current_upper_value = upper_values_list[jdx]

                adjustment = current_lower_value * current_upper_value

                previous_value = row_adjacent[current_row].get(current_column, 0.0)
                updated_value = previous_value - adjustment

                if abs(updated_value) > 1e-15:
                    row_adjacent[current_row][current_column] = updated_value
                    column_adjacent[current_column][current_row] = updated_value
                elif current_column in row_adjacent[current_row]:
                    del row_adjacent[current_row][current_column]
                    del column_adjacent[current_column][current_row]

    def convert_to_csc_format(triplets, matrix_size):
        triplets.sort(key=lambda element: (element[1], element[0]))

        values_data = [element[2] for element in triplets]
        row_indices_data = [element[0] for element in triplets]
        column_pointers = [0] * (matrix_size + 1)

        current_column = 0
        for _, col_idx, _ in triplets:
            while current_column < col_idx:
                current_column += 1
                column_pointers[current_column + 1] = column_pointers[current_column]
            column_pointers[current_column + 1] += 1

        while current_column < matrix_size - 1:
            current_column += 1
            column_pointers[current_column + 1] = column_pointers[current_column]

        return CSCMatrix(values_data, row_indices_data, column_pointers, (matrix_size, matrix_size))

    return (convert_to_csc_format(L_triplets_collection, n), 
            convert_to_csc_format(U_triplets_collection, n))


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через LU-разложение.
    """
    decomposition_result = lu_decomposition(A)
    if decomposition_result is None:
        return None
    
    L_matrix, U_matrix = decomposition_result
    dimension = len(b)

    intermediate_vector = list(b)
    for column in range(dimension):
        if intermediate_vector[column] != 0:
            for k in range(L_matrix.indptr[column], L_matrix.indptr[column + 1]):
                row_idx = L_matrix.indices[k]
                element_value = L_matrix.data[k]
                if row_idx > column:
                    intermediate_vector[row_idx] -= element_value * intermediate_vector[column]

    solution_vector = list(intermediate_vector)
    for column in range(dimension - 1, -1, -1):
        diagonal_value = 0
        for k in range(U_matrix.indptr[column], U_matrix.indptr[column + 1]):
            if U_matrix.indices[k] == column:
                diagonal_value = U_matrix.data[k]
                break

        if diagonal_value == 0: 
            return None

        solution_vector[column] /= diagonal_value

        if solution_vector[column] != 0:
            for k in range(U_matrix.indptr[column], U_matrix.indptr[column + 1]):
                row_idx = U_matrix.indices[k]
                element_value = U_matrix.data[k]
                if row_idx < column:
                    solution_vector[row_idx] -= element_value * solution_vector[column]

    return solution_vector


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через LU-разложение.
    det(A) = det(L) * det(U)
    """
    decomposition_result = lu_decomposition(A)
    if decomposition_result is None:
        return 0.0

    _, upper_matrix = decomposition_result
    determinant_value = 1
    n_dim = upper_matrix.shape[0]

    for column in range(n_dim):
        diagonal_found_flag = False
        for k in range(upper_matrix.indptr[column], upper_matrix.indptr[column + 1]):
            if upper_matrix.indices[k] == column:
                determinant_value *= upper_matrix.data[k]
                diagonal_found_flag = True
                break
        if not diagonal_found_flag:
            return 0.0

    return determinant_value
