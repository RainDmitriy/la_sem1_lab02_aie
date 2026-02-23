from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.values = data
        self.row_indices = row
        self.col_indices = col
        self.matrix_shape = shape

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу"""
        rows_count, cols_count = self.matrix_shape
        dense_matrix = [[0] * cols_count for _ in range(rows_count)]
        
        elements_count = len(self.row_indices)
        for position in range(elements_count):
            column = self.col_indices[position]
            row = self.row_indices[position]
            value = self.values[position]
            dense_matrix[row][column] = value
            
        return dense_matrix

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц"""
        all_rows = self.row_indices + other.row_indices
        all_columns = self.col_indices + other.col_indices
        all_values = self.values + other.values
        
        merged_elements = dict()
        
        for row, column, value in zip(all_rows, all_columns, all_values):
            key = (row, column)
            merged_elements[key] = merged_elements.get(key, 0) + value
            
        result_rows, result_columns, result_values = [], [], []
        
        for (row, column), value in sorted(merged_elements.items()):
            if value != 0:
                result_rows.append(row)
                result_columns.append(column)
                result_values.append(value)
                
        return COOMatrix(result_values, result_rows, result_columns, self.matrix_shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр"""
        if scalar == 0:
            return COOMatrix([], [], [], self.matrix_shape)
            
        scaled_values = [element * scalar for element in self.values]
        return COOMatrix(scaled_values, self.row_indices, self.col_indices, self.matrix_shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO"""
        triplets = []
        
        for row, column, value in zip(self.row_indices, self.col_indices, self.values):
            triplets.append([column, row, value])
            
        triplets.sort(key=lambda element: element[0])
        
        transposed_rows, transposed_columns, transposed_values = [], [], []
        transposed_shape = (self.matrix_shape[1], self.matrix_shape[0])
        
        for row, column, value in triplets:
            transposed_rows.append(row)
            transposed_columns.append(column)
            transposed_values.append(value)
            
        return COOMatrix(transposed_values, transposed_rows, transposed_columns, transposed_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц"""
        result_rows_count = self.matrix_shape[0]
        result_columns_count = other.matrix_shape[1]
        
        result_rows, result_columns, result_values = [], [], []
        result_shape = (result_rows_count, result_columns_count)
        
        product_elements = dict()
        
        for row_self, column_self, value_self in zip(self.row_indices, self.col_indices, self.values):
            for row_other, column_other, value_other in zip(other.row_indices, other.col_indices, other.values):
                if column_self == row_other:
                    key = (row_self, column_other)
                    product_elements[key] = product_elements.get(key, 0) + value_self * value_other
                    
        for (row, column), value in product_elements.items():
            if value != 0:
                result_rows.append(row)
                result_columns.append(column)
                result_values.append(value)
                
        return COOMatrix(result_values, result_rows, result_columns, result_shape)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы"""
        rows_count = len(dense_matrix)
        columns_count = len(dense_matrix[0])
        matrix_shape = (rows_count, columns_count)
        
        rows, columns, values = [], [], []
        
        for row_index in range(rows_count):
            for column_index in range(columns_count):
                current_value = dense_matrix[row_index][column_index]
                if current_value != 0:
                    rows.append(row_index)
                    columns.append(column_index)
                    values.append(current_value)
                    
        return COOMatrix(values, rows, columns, matrix_shape)

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COO в CSC
        """
        from CSC import CSCMatrix
        
        elements_to_sort = list(zip(self.row_indices, self.col_indices, self.values))
        elements_to_sort.sort(key=lambda element: element[1])
        
        if elements_to_sort:
            sorted_rows, sorted_columns, sorted_values = zip(*elements_to_sort)
        else:
            sorted_rows, sorted_columns, sorted_values = [], [], []
            
        csc_data = list(sorted_values)
        csc_indices = list(sorted_rows)
        
        columns_count = self.matrix_shape[1]
        csc_indptr = [0] * (columns_count + 1)
        
        for column_index in sorted_columns:
            csc_indptr[column_index + 1] += 1
            
        for column_position in range(columns_count):
            csc_indptr[column_position + 1] += csc_indptr[column_position]
            
        return CSCMatrix(csc_data, csc_indices, csc_indptr, self.matrix_shape)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COO в CSR
        """
        from CSR import CSRMatrix
        
        elements_to_sort = list(zip(self.row_indices, self.col_indices, self.values))
        elements_to_sort.sort()
        
        if elements_to_sort:
            sorted_rows, sorted_columns, sorted_values = zip(*elements_to_sort)
        else:
            sorted_rows, sorted_columns, sorted_values = [], [], []
            
        csr_data = list(sorted_values)
        csr_indices = list(sorted_columns)
        
        rows_count = self.matrix_shape[0]
        csr_indptr = [0] * (rows_count + 1)
        
        for row_index in sorted_rows:
            csr_indptr[row_index + 1] += 1
            
        for row_position in range(rows_count):
            csr_indptr[row_position + 1] += csr_indptr[row_position]
            
        return CSRMatrix(csr_data, csr_indices, csr_indptr, self.matrix_shape)