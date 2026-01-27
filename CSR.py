from base import Matrix
from types import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix
from CSC import CSCMatrix
from COO import COOMatrix
from typing import List


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)

        if len(indptr) != shape[0] + 1:
            raise ValueError(f"indptr должен иметь длину {shape[0] + 1}")
        if indptr[0] != 0:
            raise ValueError("indptr[0] должен быть равен 0")
        if indptr[-1] != len(data):
            raise ValueError(f"indptr[-1] должен быть равен {len(data)}")
        if len(data) != len(indices):
            raise ValueError("Длины data и indices должны совпадать")
        
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        rows, cols = self.shape

        # пустая матрица
        dense = [[0] * cols for _ in range(rows)]
        
        for i in range(rows):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            # проходим по всем элементам в строке i
            for idx in range(start, end):

                # находим столбец и значение
                j = self.indices[idx]
                value = self.data[idx]
                dense[i][j] = value
        
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""

        # будем складывать в COO
        self_coo = self._to_coo()
        other_coo = other._to_coo()
        
        result_coo = self_coo._add_impl(other_coo)
        
        # возвращаем в CSR
        return result_coo._to_csr()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""

        # Если скаляр 0, матрица пустая
        if scalar == 0:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)
        
        new_data = [value * scalar for value in self.data]

        return CSRMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Hint:
        Результат - в CSC формате (с теми же данными, но с интерпретацией столбцов как строк).
        """
        
        rows, cols = self.shape
        new_rows, new_cols = cols, rows
        
        # считаем кол-во элементов в столбцах результата
        col_counts = [0] * new_cols
        
        for i in range(rows):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            col_counts[i] = end - start
        
        # строим indptr
        new_indptr = [0] * (new_cols + 1)
        for j in range(new_cols):
            new_indptr[j + 1] = new_indptr[j] + col_counts[j]
        
        # ставим элементы в новые позиции
        new_data = [0] * len(self.data)
        new_indices = [0] * len(self.indices)
        
        # текущая позиция в каждом столюце
        col_positions = new_indptr.copy()
        
        # проходим по всем строкам начальной матрицы
        for i in range(rows):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            
            for idx in range(start, end):
                j = self.indices[idx]
                value = self.data[idx]
                
                # получаем новую позицию
                pos = col_positions[i]
                new_data[pos] = value
                new_indices[pos] = j
                col_positions[i] += 1
        
        return CSCMatrix(new_data, new_indices, new_indptr, (new_rows, new_cols))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        rows_A, cols_A = self.shape
        rows_B, cols_B = other.shape
        
        # вторая матрица будет в CSC для удобства
        B_csc = other._to_csc()
        
        # пустая матрица для результата
        result_data = []
        result_indices = []
        result_indptr = [0] * (rows_A + 1)
        
        for i in range(rows_A):
            # диапазон элементов в строке i
            row_start = self.indptr[i]
            row_end = self.indptr[i + 1]
            
            if row_start == row_end:
                result_indptr[i + 1] = result_indptr[i]
                continue
            
            # словарь для значений строки результата
            row_values = {}
            
            for a_idx in range(row_start, row_end):
                k = self.indices[a_idx]  # столбец в A
                a_val = self.data[a_idx]
                
                # соответствующий столбец k в матрице B
                col_start = B_csc.indptr[k]
                col_end = B_csc.indptr[k + 1]
                
                # умножаем на все элементы в столбце k матрицы B
                for b_idx in range(col_start, col_end):
                    j = B_csc.indices[b_idx]  # строка в B
                    b_val = B_csc.data[b_idx]
                    
                    # добавляем в результат
                    if j not in row_values:
                        row_values[j] = 0
                    row_values[j] += a_val * b_val
            
            # сортировка столбцов
            # добавляем ненулевые элементы в результат
            sorted_cols = sorted(row_values.keys())
            for j in sorted_cols:
                value = row_values[j]
                if value != 0:
                    result_data.append(value)
                    result_indices.append(j)
            
            # новый указатель на конец строки
            result_indptr[i + 1] = len(result_data)
        
        return CSRMatrix(result_data, result_indices, result_indptr, (rows_A, cols_B))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0])
        
        data = []
        indices = []
        
        # считаем элементы в строках
        row_counts = [0] * rows
        
        for i in range(rows):
            for j in range(cols):
                value = dense_matrix[i][j]
                if value != 0:
                    data.append(value)
                    indices.append(j)
                    row_counts[i] += 1
        
        # строим indptr
        indptr = [0] * (rows + 1)
        for i in range(rows):
            indptr[i + 1] = indptr[i] + row_counts[i]
        
        return cls(data, indices, indptr, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSRMatrix в CSCMatrix.
        """
        # чтобы преобразовать в CSC, транспонируем
        return self.transpose()
    
    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSRMatrix в COOMatrix.
        """
        rows, cols = self.shape

        # пустой COO шаблон
        data = []
        row_indices = []
        col_indices = []
        
        for i in range(rows):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            
            # для кажноо элементта в каждой строке находим столбец и значение
            for idx in range(start, end):
                j = self.indices[idx]
                value = self.data[idx]
                
                # добавляем
                data.append(value)
                row_indices.append(i)
                col_indices.append(j)
        
        return COOMatrix(data, row_indices, col_indices, self.shape)