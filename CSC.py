from base import Matrix
from types import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix
from CSR import CSRMatrix
from COO import COOMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)

        if len(indptr) != shape[1] + 1:
            raise ValueError(f"indptr должен иметь длину {shape[1] + 1}")
        if indptr[0] != 0:
            raise ValueError("indptr[0] должен быть 0")
        if indptr[-1] != len(data):
            raise ValueError(f"indptr[-1] должен быть равен {len(data)}")
        if len(data) != len(indices):
            raise ValueError(f"длины data и indices должны быть равны")
        
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу."""
        rows, cols = self.shape

        # делаем шаблон с нулями
        dense = [[0] * cols for _ in range(rows)]
        
        # в каждом столбце проходимся по всем его элементам
        for j in range(cols):
            start = self.indptr[j]
            end = self.indptr[j + 1]

            for idx in range(start, end):
                i = self.indices[idx]
                value = self.data[idx]
                dense[i][j] = value
        
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        # складыввать проще в COO формате
        self_coo = self._to_coo()
        other_coo = other._to_coo()
        result_coo = self_coo._add_impl(other_coo)
        
        # преобразуем в CSC
        return result_coo._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""

        # если скаляр 0, получаеся пустая матрица
        if scalar == 0:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)
        
        # просто умножаем все значения на скаляр
        new_data = [value * scalar for value in self.data]
        
        # обновляются только значения
        return CSCMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Hint:
        Результат - в CSR формате (с теми же данными, но с интерпретацией строк как столбцов).
        """
        rows, cols = self.shape
        new_rows, new_cols = cols, rows
        
        # считаем кол-во элементов в каждой строке результата
        row_counts = [0] * new_rows
        
        for j in range(cols):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            row_counts[j] = end - start
        
        # cтроим indptr
        new_indptr = [0] * (new_rows + 1)
        for i in range(new_rows):
            new_indptr[i + 1] = new_indptr[i] + row_counts[i]
        
        # ставим элементы на новые позиции
        new_data = [0] * len(self.data)
        new_indices = [0] * len(self.indices)
        
        row_positions = new_indptr.copy()
        
        # проходим по столбцам начальной матрицы
        for j in range(cols):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            
            for idx in range(start, end):
                i = self.indices[idx]
                value = self.data[idx]
                
                pos = row_positions[j]  # j теперь строка
                new_data[pos] = value
                new_indices[pos] = i  # i теперь столбец
                row_positions[j] += 1
        
        return CSRMatrix(new_data, new_indices, new_indptr, (new_rows, new_cols))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        
        rows_A, cols_A = self.shape
        rows_B, cols_B = other.shape
        
        # первая матрица будет в CSR, чтобы удобно брать строки
        A_csr = self._to_csr()

        # переименую вторую, чтобы не запутаться
        B_csc = other
        
        # результат будет в COO для удобства
        result_data = []
        result_row = []
        result_col = []
        
        for i in range(rows_A):
            
            # диапазон элементов в строке i
            row_start = A_csr.indptr[i]
            row_end = A_csr.indptr[i + 1]
            
            row_result = {}
            
            for a_idx in range(row_start, row_end):
                k = A_csr.indices[a_idx]  # столбец A
                a_val = A_csr.data[a_idx]
                
                # соответствующий столбец k в матрице B
                col_start = B_csc.indptr[k]
                col_end = B_csc.indptr[k + 1]
                
                for b_idx in range(col_start, col_end):
                    j = B_csc.indices[b_idx]  # строка B
                    b_val = B_csc.data[b_idx]
                    
                    # вставляем в результат
                    if j not in row_result:
                        row_result[j] = 0
                    row_result[j] += a_val * b_val
            
            # все ненулевые строки добавляем в результат
            for j, value in row_result.items():
                if value != 0:
                    result_data.append(value)
                    result_row.append(i)
                    result_col.append(j)
        
        result_coo = COOMatrix(result_data, result_row, result_col, (rows_A, cols_B))
        return result_coo._to_csc()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0])
        
        data = []
        indices = []
        
        # считаем элементы в столбцах
        col_counts = [0] * cols
        
        for j in range(cols):
            for i in range(rows):
                value = dense_matrix[i][j]
                if value != 0:
                    data.append(value)
                    indices.append(i)
                    col_counts[j] += 1
        
        # строим indptr
        indptr = [0] * (cols + 1)
        for j in range(cols):
            indptr[j + 1] = indptr[j] + col_counts[j]
        
        return CSCMatrix(data, indices, indptr, (rows, cols))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        # чтобы преобразовать в CSR, просто транспонируем
        return self.transpose()

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        rows, cols = self.shape

        # пустой шаблон
        data = []
        row_indices = []
        col_indices = []
        
        # проходимся по столбцам
        for j in range(cols):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            
            # для каждого элемента получаем строку и значение
            for idx in range(start, end):
                i = self.indices[idx]
                value = self.data[idx]
                
                data.append(value)
                row_indices.append(i)
                col_indices.append(j)
        
        return COOMatrix(data, row_indices, col_indices, self.shape)
