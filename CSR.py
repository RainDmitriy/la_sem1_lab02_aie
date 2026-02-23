from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.values = data
        self.column_indices = indices
        self.row_pointer = indptr
        self.matrix_shape = shape

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу"""
        rows_count, columns_count = self.matrix_shape
        dense_matrix = [[0] * columns_count for _ in range(rows_count)]
        
        element_position = 0
        current_row = 0
        
        while current_row < len(self.row_pointer) - 1:
            elements_in_row = self.row_pointer[current_row + 1] - self.row_pointer[current_row]
            
            for _ in range(elements_in_row):
                column_index = self.column_indices[element_position]
                dense_matrix[current_row][column_index] = self.values[element_position]
                element_position += 1
                
            current_row += 1
            
        return dense_matrix

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц"""
        result_row_pointer = [0]
        result_column_indices = []
        result_values = []
        
        position_self = 0
        position_other = 0
        current_row = 0
        
        while current_row < len(self.row_pointer) - 1:
            merged_elements = dict()
            
            elements_self = self.row_pointer[current_row + 1] - self.row_pointer[current_row]
            for _ in range(elements_self):
                column = self.column_indices[position_self]
                merged_elements[column] = merged_elements.get(column, 0) + self.values[position_self]
                position_self += 1
                
            elements_other = other.row_pointer[current_row + 1] - other.row_pointer[current_row]
            for _ in range(elements_other):
                column = other.column_indices[position_other]
                merged_elements[column] = merged_elements.get(column, 0) + other.values[position_other]
                position_other += 1
                
            added_elements_count = 0
            
            for column, value in sorted(merged_elements.items()):
                if value != 0:
                    result_column_indices.append(column)
                    result_values.append(value)
                    added_elements_count += 1
                    
            result_row_pointer.append(result_row_pointer[-1] + added_elements_count)
            current_row += 1
            
        return CSRMatrix(result_values, result_column_indices, result_row_pointer, self.matrix_shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр"""
        if scalar == 0:
            return CSRMatrix([], [], [0] * len(self.row_pointer), self.matrix_shape)
            
        scaled_values = [element * scalar for element in self.values]
        return CSRMatrix(scaled_values, self.column_indices, self.row_pointer, self.matrix_shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Получаем в CSC формате(с теми же данными, но с интерпретацией столбцов как строк)
        """
        from CSC import CSCMatrix
        return CSCMatrix(self.values, self.column_indices, self.row_pointer, 
                        (self.matrix_shape[1], self.matrix_shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц"""
        rows_count_A = self.matrix_shape[0]
        columns_count_B = other.matrix_shape[1]
        
        result_row_pointer = [0]
        result_column_indices = []
        result_values = []
        
        for row_index in range(rows_count_A):
            row_sum = {}
            
            for position in range(self.row_pointer[row_index], self.row_pointer[row_index + 1]):
                column_A = self.column_indices[position]
                value_A = self.values[position]
                
                start_B = other.row_pointer[column_A]
                end_B = other.row_pointer[column_A + 1]
                
                for inner_position in range(start_B, end_B):
                    column_B = other.column_indices[inner_position]
                    value_B = other.values[inner_position]
                    
                    product = value_A * value_B
                    if product != 0:
                        row_sum[column_B] = row_sum.get(column_B, 0) + product
                        
            if row_sum:
                sorted_items = sorted(row_sum.items())
                for column, value in sorted_items:
                    if value != 0:
                        result_column_indices.append(column)
                        result_values.append(value)
                        
            result_row_pointer.append(len(result_values))
            
        return CSRMatrix(result_values, result_column_indices, result_row_pointer, 
                        (rows_count_A, columns_count_B))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы"""
        row_pointer = [0]
        column_indices = []
        values = []
        
        rows_count = len(dense_matrix)
        columns_count = len(dense_matrix[0])
        
        for row_index in range(rows_count):
            current_row_indices = []
            
            for column_index in range(columns_count):
                element = dense_matrix[row_index][column_index]
                if element != 0:
                    values.append(element)
                    current_row_indices.append(column_index)
                    
            row_pointer.append(len(current_row_indices) + row_pointer[-1])
            column_indices.extend(current_row_indices)
            
        return CSRMatrix(values, column_indices, row_pointer, (rows_count, columns_count))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSR в CSC
        """
        from CSC import CSCMatrix
        
        rows_count, columns_count = self.matrix_shape
        nonzero_count = len(self.values)
        
        elements_per_column = [0] * columns_count
        for column in self.column_indices:
            elements_per_column[column] += 1
            
        csc_column_pointer = [0] * (columns_count + 1)
        cumulative_sum = 0
        
        for column_index in range(columns_count):
            csc_column_pointer[column_index] = cumulative_sum
            cumulative_sum += elements_per_column[column_index]
        csc_column_pointer[columns_count] = cumulative_sum
        
        working_pointer = csc_column_pointer[:-1].copy()
        
        csc_row_indices = [0] * nonzero_count
        csc_values = [0] * nonzero_count
        
        for row_index in range(rows_count):
            for position in range(self.row_pointer[row_index], self.row_pointer[row_index + 1]):
                column_index = self.column_indices[position]
                value = self.values[position]
                
                destination = working_pointer[column_index]
                csc_row_indices[destination] = row_index
                csc_values[destination] = value
                working_pointer[column_index] += 1
                
        return CSCMatrix(csc_values, csc_row_indices, csc_column_pointer, self.matrix_shape)

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSR в COO
        """
        from COO import COOMatrix
        
        coo_columns = self.column_indices.copy()
        coo_rows = []
        
        rows_count = self.matrix_shape[0]
        
        for row_index in range(rows_count):
            elements_in_row = self.row_pointer[row_index + 1] - self.row_pointer[row_index]
            coo_rows.extend([row_index] * elements_in_row)
            
        return COOMatrix(self.values, coo_rows, coo_columns, self.matrix_shape)