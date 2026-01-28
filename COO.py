from typing import List, Tuple
from base import Matrix
from type import DenseMatrix, Shape, Vector, COOData, COORows, COOCols, EPS


class COOMatrix(Matrix):
    """Класс для хранения матрицы в формате COO (Coordinate)"""
    
    def __init__(self, rows: COORows, cols: COOCols, data: COOData, shape: Shape):
        super().__init__(shape)
        self.rows = rows
        self.cols = cols
        self.data = data
        self.nnz = len(data)
        
        # Проверка согласованности
        if len(rows) != len(cols) or len(rows) != len(data):
            raise ValueError("Длины rows, cols и data должны совпадать")
        
        # Проверка границ
        n_rows, n_cols = shape
        for row, col in zip(rows, cols):
            if row < 0 or row >= n_rows or col < 0 or col >= n_cols:
                raise ValueError(f"Индекс ({row}, {col}) выходит за границы матрицы {shape}")
    
    def to_dense(self) -> DenseMatrix:
        n_rows, n_cols = self.shape
        dense = [[0.0] * n_cols for _ in range(n_rows)]
        
        for row, col, val in zip(self.rows, self.cols, self.data):
            dense[row][col] = val
        
        return dense
    
    @classmethod
    def from_dense(cls, matrix: DenseMatrix) -> 'COOMatrix':
        rows = []
        cols = []
        data = []
        
        n_rows = len(matrix)
        n_cols = len(matrix[0]) if n_rows > 0 else 0
        
        for i in range(n_rows):
            for j in range(n_cols):
                if abs(matrix[i][j]) > EPS:
                    rows.append(i)
                    cols.append(j)
                    data.append(float(matrix[i][j]))
        
        return cls(rows, cols, data, (n_rows, n_cols))
    
    def to_csr(self) -> 'CSRMatrix':
        """Конвертация COO в CSR"""
        from CSR import CSRMatrix
        
        # Сортируем элементы по строкам, затем по столбцам
        elements = list(zip(self.rows, self.cols, self.data))
        elements.sort(key=lambda x: (x[0], x[1]))
        
        n_rows, n_cols = self.shape
        data = []
        indices = []
        indptr = [0]
        
        current_row = -1
        for row, col, val in elements:
            # Заполняем indptr для пропущенных строк
            while current_row < row:
                indptr.append(len(data))
                current_row += 1
            
            data.append(val)
            indices.append(col)
        
        # Завершаем indptr
        while current_row < n_rows - 1:
            indptr.append(len(data))
            current_row += 1
        indptr.append(len(data))
        
        return CSRMatrix(data, indices, indptr, self.shape)
    
    def to_csc(self) -> 'CSCMatrix':
        """Конвертация COO в CSC."""
        # Через транспонирование
        transposed = self.transpose()
        from CSC import CSCMatrix
        return CSCMatrix(transposed.cols, transposed.rows, transposed.data, self.shape)
    
    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        # Конвертируем обе матрицы в dense и обратно
        dense_self = self.to_dense()
        dense_other = other.to_dense()
        
        n_rows, n_cols = self.shape
        result_dense = [[0.0] * n_cols for _ in range(n_rows)]
        
        for i in range(n_rows):
            for j in range(n_cols):
                result_dense[i][j] = dense_self[i][j] + dense_other[i][j]
        
        return COOMatrix.from_dense(result_dense)
    
    def _mul_impl(self, scalar: float) -> 'Matrix':
        new_data = [val * scalar for val in self.data]
        return COOMatrix(self.rows.copy(), self.cols.copy(), new_data, self.shape)
    
    def transpose(self) -> 'Matrix':
        return COOMatrix(self.cols, self.rows, self.data, (self.shape[1], self.shape[0]))
    
    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        # Простая реализация через dense
        dense_self = self.to_dense()
        dense_other = other.to_dense()
        
        n_rows, n_cols_self = self.shape
        n_cols_other = other.shape[1]
        result_dense = [[0.0] * n_cols_other for _ in range(n_rows)]
        
        for i in range(n_rows):
            for j in range(n_cols_other):
                sum_val = 0.0
                for k in range(n_cols_self):
                    sum_val += dense_self[i][k] * dense_other[k][j]
                result_dense[i][j] = sum_val
        
        return COOMatrix.from_dense(result_dense)
    
    def __str__(self) -> str:
        return f"COOMatrix(shape={self.shape}, nnz={self.nnz})"