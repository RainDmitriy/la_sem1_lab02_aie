from base import Matrix
from types import COOData, COORows, COOCols, Shape, DenseMatrix
from CSC import CSCMatrix
from CSR import CSRMatrix
from typing import Dict, Tuple, List


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)

        # длины должны быть равны
        if not (len(data) == len(row) == len(col)):
            raise ValueError("Длины data, row, col не равны")
        
        self.data = data
        self.row = row
        self.col = col
        self.shape = shape

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу."""
        rows, cols = self.shape
        dense = []

        # делаем загготовку для матрицы
        for i in range(rows):
            row = [0] * cols
            dense.append(row)

        # заполняем
        for i in range(len(self.data)):
            dense[ self.row[i] ][ self.col[i] ] = self.data[i]
    
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц."""
        
        # словарь для сложения значений
        sum_dict: Dict[Tuple[int, int], float] = {}
        
        # первая матрица
        for value, row, col in zip(self.data, self.row, self.col):
            sum_dict[(row, col)] = sum_dict.get((row, col), 0) + value
        
        # вторая матрица
        for value, row, col in zip(other.data, other.row, other.col):
            sum_dict[(row, col)] = sum_dict.get((row, col), 0) + value
        
        # составляем новую матрицу
        new_data, new_row, new_col = [], [], []
        for (row, col), value in sum_dict.items():
            if value != 0:
                new_data.append(value)
                new_row.append(row)
                new_col.append(col)
        
        return COOMatrix(new_data, new_row, new_col, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        for i in range(len(self.data)):
            self.data[i] *= scalar
        return self

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        new_shape = (self.shape[1], self.shape[0])  # меняем местами строки и столбцы
        return COOMatrix(self.data.copy(), self.col.copy(), self.row.copy(), new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""
        
        # преобразуем вторую матрицу в CSC для удобвства
        other_csc = other._to_csc()
        
        rows_A, _ = self.shape
        _, cols_B = other.shape
        # cols_A и rows_B не приодятся
        
        # для созранения результата делаем временный словарь словарей
        result_dict: Dict[int, Dict[int, float]] = {i: {} for i in range(rows_A)}
        
        # каждый A_{i,k} умножаем на все B_{k,j}
        for a_val, i, k in zip(self.data, self.row, self.col):
            col_start = other_csc.indptr[k]
            col_end = other_csc.indptr[k + 1]
            
            for idx in range(col_start, col_end):
                j = other_csc.indices[idx]
                b_val = other_csc.data[idx]
                
                # добавляем к результату в i,j
                if j not in result_dict[i]:
                    result_dict[i][j] = 0
                result_dict[i][j] += a_val * b_val
        
        # результат должен быть COO
        result_data, result_row, result_col = [], [], []
        for i in range(rows_A):
            for j, val in result_dict[i].items():
                if val != 0:
                    result_data.append(val)
                    result_row.append(i)
                    result_col.append(j)
        
        return COOMatrix(result_data, result_row, result_col, (rows_A, cols_B))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        data, row, col = [], [], []

        for i in range(len(dense_matrix)):
            for j in range(len(dense_matrix[0])):
                value = dense_matrix[i][j]
                if value != 0:
                    data.append(value)
                    row.append(i)
                    col.append(j)

        return COOMatrix(data, row, col, (len(dense_matrix), len(dense_matrix[0])))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        rows, cols = self.shape
        
        # для сортировки по столбцам и строкам нужен список кортежей
        elements = list(zip(self.col, self.row, self.data))
        elements.sort()
        
        # списки пустой CSC матрицы
        data: List[float] = []
        indices: List[int] = []
        indptr: List[int] = [0] * (cols + 1)
        
        # заполняем data и indices
        for col_idx, row_idx, value in elements:
            data.append(value)
            indices.append(row_idx)
            indptr[col_idx + 1] += 1
        
        for j in range(cols):
            indptr[j + 1] += indptr[j]
        
        return CSCMatrix(data, indices, indptr, self.shape)


    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """

        # тут так же, как в CSC, только сортировка сначала по строкам
        rows, cols = self.shape
        
        # список кортежей, чтобы отсортировать по строкам и столбцам
        elements = list(zip(self.row, self.col, self.data))
        elements.sort()
        
        # списки пустой CSR матрицы
        data: List[float] = []
        indices: List[int] = []
        indptr: List[int] = [0] * (rows + 1)
        
        # заполняем data и indices
        for row_idx, col_idx, value in elements:
            data.append(value)
            indices.append(col_idx)
            indptr[row_idx + 1] += 1
        
        for i in range(rows):
            indptr[i + 1] += indptr[i]
        
        return CSRMatrix(data, indices, indptr, self.shape)
