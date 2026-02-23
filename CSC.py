from base import Matrix
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.shape = shape

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу"""
        rows_count, columns_count = self.shape
        dense_matrix = [[0] * columns_count for _ in range(rows_count)]
        
        element_position = 0
        current_column = 0
        
        while current_column < len(self.indptr) - 1:
            elements_in_column = self.indptr[current_column + 1] - self.indptr[current_column]
            
            for _ in range(elements_in_column):
                row_index = self.indices[element_position]
                dense_matrix[row_index][current_column] = self.data[element_position]
                element_position += 1
                
            current_column += 1
            
        return dense_matrix

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц"""
        result_indptr = [0]
        result_indices = []
        result_data = []
        
        position_self = 0
        position_other = 0
        current_column = 0
        
        while current_column < len(self.indptr) - 1:
            merged_elements = dict()
            
            elements_self = self.indptr[current_column + 1] - self.indptr[current_column]
            for _ in range(elements_self):
                row = self.indices[position_self]
                merged_elements[row] = merged_elements.get(row, 0) + self.data[position_self]
                position_self += 1
                
            elements_other = other.indptr[current_column + 1] - other.indptr[current_column]
            for _ in range(elements_other):
                row = other.indices[position_other]
                merged_elements[row] = merged_elements.get(row, 0) + other.data[position_other]
                position_other += 1
                
            added_elements_count = 0
            
            if merged_elements:
                for row, value in sorted(merged_elements.items()):
                    if value != 0:
                        result_indices.append(row)
                        result_data.append(value)
                        added_elements_count += 1
                        
            result_indptr.append(result_indptr[-1] + added_elements_count)
            current_column += 1
            
        return CSCMatrix(result_data, result_indices, result_indptr, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр"""
        if scalar == 0:
            return CSCMatrix([], [], [0] * len(self.indptr), self.shape)
            
        scaled_data = [element * scalar for element in self.data]
        return CSCMatrix(scaled_data, self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы
        Получаем в CSR формате(с теми же данными, но с интерпретацией строк как столбцов)
        """
        from CSR import CSRMatrix
        return CSRMatrix(self.data, self.indices, self.indptr, 
                        (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц"""
        csr_self = self._to_csr()
        csr_other = other._to_csr()
        csr_result = csr_self._matmul_impl(csr_other)
        return csr_result._to_csc()
    
    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы"""
        indptr = [0]
        indices = []
        data = []
        
        rows_count = len(dense_matrix)
        columns_count = len(dense_matrix[0])
        
        for column_index in range(rows_count):
            current_column_indices = []
            
            for row_index in range(columns_count):
                element = dense_matrix[row_index][column_index]
                if element != 0:
                    data.append(element)
                    current_column_indices.append(row_index)
                    
            indptr.append(len(current_column_indices) + indptr[-1])
            indices.extend(current_column_indices)
            
        return CSCMatrix(data, indices, indptr, (rows_count, columns_count))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSC в CSR
        """
        from CSR import CSRMatrix
        
        rows_count, columns_count = self.shape
        nonzero_count = len(self.data)
        
        elements_per_row = [0] * rows_count
        for row in self.indices:
            elements_per_row[row] += 1
            
        csr_indptr = [0] * (rows_count + 1)
        cumulative_sum = 0
        
        for row_index in range(rows_count):
            csr_indptr[row_index] = cumulative_sum
            cumulative_sum += elements_per_row[row_index]
        csr_indptr[rows_count] = cumulative_sum
        
        working_pointer = csr_indptr[:-1].copy()
        
        csr_indices = [0] * nonzero_count
        csr_data = [0] * nonzero_count
        
        for column_index in range(columns_count):
            for position in range(self.indptr[column_index], self.indptr[column_index + 1]):
                row_index = self.indices[position]
                value = self.data[position]
                
                destination = working_pointer[row_index]
                csr_indices[destination] = column_index
                csr_data[destination] = value
                working_pointer[row_index] += 1
                
        return CSRMatrix(csr_data, csr_indices, csr_indptr, self.shape)

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSC в COO
        """
        from COO import COOMatrix
        
        coo_rows = self.indices.copy()
        coo_columns = []
        
        columns_count = self.shape[1]
        
        for column_index in range(columns_count):
            elements_in_column = self.indptr[column_index + 1] - self.indptr[column_index]
            coo_columns.extend([column_index] * elements_in_column)
            
        return COOMatrix(self.data, coo_rows, coo_columns, self.shape)