from base import Matrix
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.values = data
        self.row_indices = indices
        self.column_pointer = indptr
        self.matrix_shape = shape

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу"""
        rows_count, columns_count = self.matrix_shape
        dense_matrix = [[0] * columns_count for _ in range(rows_count)]
        
        element_position = 0
        current_column = 0
        
        while current_column < len(self.column_pointer) - 1:
            elements_in_column = self.column_pointer[current_column + 1] - self.column_pointer[current_column]
            
            for _ in range(elements_in_column):
                row_index = self.row_indices[element_position]
                dense_matrix[row_index][current_column] = self.values[element_position]
                element_position += 1
                
            current_column += 1
            
        return dense_matrix

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц"""
        result_column_pointer = [0]
        result_row_indices = []
        result_values = []
        
        position_self = 0
        position_other = 0
        current_column = 0
        
        while current_column < len(self.column_pointer) - 1:
            merged_elements = dict()
            
            elements_self = self.column_pointer[current_column + 1] - self.column_pointer[current_column]
            for _ in range(elements_self):
                row = self.row_indices[position_self]
                merged_elements[row] = merged_elements.get(row, 0) + self.values[position_self]
                position_self += 1
                
            elements_other = other.column_pointer[current_column + 1] - other.column_pointer[current_column]
            for _ in range(elements_other):
                row = other.row_indices[position_other]
                merged_elements[row] = merged_elements.get(row, 0) + other.values[position_other]
                position_other += 1
                
            added_elements_count = 0
            
            if merged_elements:
                for row, value in sorted(merged_elements.items()):
                    if value != 0:
                        result_row_indices.append(row)
                        result_values.append(value)
                        added_elements_count += 1
                        
            result_column_pointer.append(result_column_pointer[-1] + added_elements_count)
            current_column += 1
            
        return CSCMatrix(result_values, result_row_indices, result_column_pointer, self.matrix_shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр"""
        if scalar == 0:
            return CSCMatrix([], [], [0] * len(self.column_pointer), self.matrix_shape)
            
        scaled_values = [element * scalar for element in self.values]
        return CSCMatrix(scaled_values, self.row_indices, self.column_pointer, self.matrix_shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы
        Получаем в CSR формате(с теми же данными, но с интерпретацией строк как столбцов)
        """
        from CSR import CSRMatrix
        return CSRMatrix(self.values, self.row_indices, self.column_pointer, 
                        (self.matrix_shape[1], self.matrix_shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц"""
        csr_self = self._to_csr()
        csr_other = other._to_csr()
        csr_result = csr_self._matmul_impl(csr_other)
        return csr_result._to_csc()
    
    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы"""
        column_pointer = [0]
        row_indices = []
        values = []
        
        rows_count = len(dense_matrix)
        columns_count = len(dense_matrix[0])
        
        for column_index in range(rows_count):
            current_column_indices = []
            
            for row_index in range(columns_count):
                element = dense_matrix[row_index][column_index]
                if element != 0:
                    values.append(element)
                    current_column_indices.append(row_index)
                    
            column_pointer.append(len(current_column_indices) + column_pointer[-1])
            row_indices.extend(current_column_indices)
            
        return CSCMatrix(values, row_indices, column_pointer, (rows_count, columns_count))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSC в CSR
        """
        from CSR import CSRMatrix
        
        rows_count, columns_count = self.matrix_shape
        nonzero_count = len(self.values)
        
        elements_per_row = [0] * rows_count
        for row in self.row_indices:
            elements_per_row[row] += 1
            
        csr_row_pointer = [0] * (rows_count + 1)
        cumulative_sum = 0
        
        for row_index in range(rows_count):
            csr_row_pointer[row_index] = cumulative_sum
            cumulative_sum += elements_per_row[row_index]
        csr_row_pointer[rows_count] = cumulative_sum
        
        working_pointer = csr_row_pointer[:-1].copy()
        
        csr_column_indices = [0] * nonzero_count
        csr_values = [0] * nonzero_count
        
        for column_index in range(columns_count):
            for position in range(self.column_pointer[column_index], self.column_pointer[column_index + 1]):
                row_index = self.row_indices[position]
                value = self.values[position]
                
                destination = working_pointer[row_index]
                csr_column_indices[destination] = column_index
                csr_values[destination] = value
                working_pointer[row_index] += 1
                
        return CSRMatrix(csr_values, csr_column_indices, csr_row_pointer, self.matrix_shape)

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSC в COO
        """
        from COO import COOMatrix
        
        coo_rows = self.row_indices.copy()
        coo_columns = []
        
        columns_count = self.matrix_shape[1]
        
        for column_index in range(columns_count):
            elements_in_column = self.column_pointer[column_index + 1] - self.column_pointer[column_index]
            coo_columns.extend([column_index] * elements_in_column)
            
        return COOMatrix(self.values, coo_rows, coo_columns, self.matrix_shape)