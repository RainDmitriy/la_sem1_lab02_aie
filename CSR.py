from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.shape = shape

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу"""
        rows_count, columns_count = self.shape
        dense_matrix = [[0] * columns_count for _ in range(rows_count)]
        
        element_position = 0
        current_row = 0
        
        while current_row < len(self.indptr) - 1:
            elements_in_row = self.indptr[current_row + 1] - self.indptr[current_row]
            
            for _ in range(elements_in_row):
                column_index = self.indices[element_position]
                dense_matrix[current_row][column_index] = self.data[element_position]
                element_position += 1
                
            current_row += 1
            
        return dense_matrix

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц"""
        result_indptr = [0]
        result_indices = []
        result_data = []
        
        position_self = 0
        position_other = 0
        current_row = 0
        
        while current_row < len(self.indptr) - 1:
            merged_elements = dict()
            
            elements_self = self.indptr[current_row + 1] - self.indptr[current_row]
            for _ in range(elements_self):
                column = self.indices[position_self]
                merged_elements[column] = merged_elements.get(column, 0) + self.data[position_self]
                position_self += 1
                
            elements_other = other.indptr[current_row + 1] - other.indptr[current_row]
            for _ in range(elements_other):
                column = other.indices[position_other]
                merged_elements[column] = merged_elements.get(column, 0) + other.data[position_other]
                position_other += 1
                
            added_elements_count = 0
            
            for column, value in sorted(merged_elements.items()):
                if value != 0:
                    result_indices.append(column)
                    result_data.append(value)
                    added_elements_count += 1
                    
            result_indptr.append(result_indptr[-1] + added_elements_count)
            current_row += 1
            
        return CSRMatrix(result_data, result_indices, result_indptr, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр"""
        if scalar == 0:
            return CSRMatrix([], [], [0] * len(self.indptr), self.shape)
            
        scaled_data = [element * scalar for element in self.data]
        return CSRMatrix(scaled_data, self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Получаем в CSC формате(с теми же данными, но с интерпретацией столбцов как строк)
        """
        from CSC import CSCMatrix
        return CSCMatrix(self.data, self.indices, self.indptr, 
                        (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц"""
        rows_count_A = self.shape[0]
        columns_count_B = other.shape[1]
        
        result_indptr = [0]
        result_indices = []
        result_data = []
        
        for row_index in range(rows_count_A):
            row_sum = {}
            
            for position in range(self.indptr[row_index], self.indptr[row_index + 1]):
                column_A = self.indices[position]
                value_A = self.data[position]
                
                start_B = other.indptr[column_A]
                end_B = other.indptr[column_A + 1]
                
                for inner_position in range(start_B, end_B):
                    column_B = other.indices[inner_position]
                    value_B = other.data[inner_position]
                    
                    product = value_A * value_B
                    if product != 0:
                        row_sum[column_B] = row_sum.get(column_B, 0) + product
                        
            if row_sum:
                sorted_items = sorted(row_sum.items())
                for column, value in sorted_items:
                    if value != 0:
                        result_indices.append(column)
                        result_data.append(value)
                        
            result_indptr.append(len(result_data))
            
        return CSRMatrix(result_data, result_indices, result_indptr, 
                        (rows_count_A, columns_count_B))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы"""
        indptr = [0]
        indices = []
        data = []
        
        rows_count = len(dense_matrix)
        columns_count = len(dense_matrix[0])
        
        for row_index in range(rows_count):
            current_row_indices = []
            
            for column_index in range(columns_count):
                element = dense_matrix[row_index][column_index]
                if element != 0:
                    data.append(element)
                    current_row_indices.append(column_index)
                    
            indptr.append(len(current_row_indices) + indptr[-1])
            indices.extend(current_row_indices)
            
        return CSRMatrix(data, indices, indptr, (rows_count, columns_count))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSR в CSC
        """
        from CSC import CSCMatrix
        
        rows_count, columns_count = self.shape
        nonzero_count = len(self.data)
        
        elements_per_column = [0] * columns_count
        for column in self.indices:
            elements_per_column[column] += 1
            
        csc_indptr = [0] * (columns_count + 1)
        cumulative_sum = 0
        
        for column_index in range(columns_count):
            csc_indptr[column_index] = cumulative_sum
            cumulative_sum += elements_per_column[column_index]
        csc_indptr[columns_count] = cumulative_sum
        
        working_pointer = csc_indptr[:-1].copy()
        
        csc_indices = [0] * nonzero_count
        csc_data = [0] * nonzero_count
        
        for row_index in range(rows_count):
            for position in range(self.indptr[row_index], self.indptr[row_index + 1]):
                column_index = self.indices[position]
                value = self.data[position]
                
                destination = working_pointer[column_index]
                csc_indices[destination] = row_index
                csc_data[destination] = value
                working_pointer[column_index] += 1
                
        return CSCMatrix(csc_data, csc_indices, csc_indptr, self.shape)

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSR в COO
        """
        from COO import COOMatrix
        
        coo_columns = self.indices.copy()
        coo_rows = []
        
        rows_count = self.shape[0]
        
        for row_index in range(rows_count):
            elements_in_row = self.indptr[row_index + 1] - self.indptr[row_index]
            coo_rows.extend([row_index] * elements_in_row)
            
        return COOMatrix(self.data, coo_rows, coo_columns, self.shape)