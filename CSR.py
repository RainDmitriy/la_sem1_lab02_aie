from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix

# без этой части у меня возникают проблемы с типизацией
# способ решения нашёл в интернете
# (в файле каждого класса подписал на всякий случай)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from CSC import CSCMatrix
    from COO import COOMatrix


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

        dense = [[0] * cols for _ in range(rows)]
        
        for i in range(rows):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for idx in range(start, end):

                j = self.indices[idx]
                value = self.data[idx]
                dense[i][j] = value
        
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""

        self_coo = self._to_coo()
        other_coo = other._to_coo()
        
        result_coo = self_coo._add_impl(other_coo)
        
        return result_coo._to_csr()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""

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
        from CSC import CSCMatrix
        
        rows, cols = self.shape
        new_rows, new_cols = cols, rows
        
        # считаем кол-во элементов в столбцах результата
        col_counts = [0] * new_cols
        
        for i in range(rows):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            col_counts[i] = end - start
        
        new_indptr = [0] * (new_cols + 1)
        for j in range(new_cols):
            new_indptr[j + 1] = new_indptr[j] + col_counts[j]
        
        new_data = [0] * len(self.data)
        new_indices = [0] * len(self.indices)
        
        col_positions = new_indptr.copy()
        
        for i in range(rows):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            
            for idx in range(start, end):
                j = self.indices[idx]
                value = self.data[idx]
                
                pos = col_positions[i]
                new_data[pos] = value
                new_indices[pos] = j
                col_positions[i] += 1
        
        return CSCMatrix(new_data, new_indices, new_indptr, (new_rows, new_cols))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        rows_A, cols_A = self.shape
        rows_B, cols_B = other.shape

        result_data = []
        result_indices = []
        result_indptr = [0] * (rows_A + 1)

        for i in range(rows_A):
            row_summ = {}

            a_start = self.indptr[i]
            a_end = self.indptr[i + 1]

            for a_idx in range(a_start, a_end):
                k = self.indices[a_idx]
                a_val = self.data[a_idx]

                b_start = other.indptr[k]
                b_end = other.indptr[k + 1]

                for b_idx in range(b_start, b_end):
                    j = other.indices[b_idx]
                    b_val = other.data[b_idx]

                    if j not in row_summ:
                        row_summ[j] = 0.0
                    row_summ[j] += a_val * b_val

            sorted_cols = sorted(row_summ.keys())
            for j in sorted_cols:
                val = row_summ[j]
                if abs(val) > 1e-14:
                    result_data.append(val)
                    result_indices.append(j)

            result_indptr[i + 1] = len(result_data)

        return CSRMatrix(result_data, result_indices, result_indptr, (rows_A, cols_B))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0])
        
        data = []
        indices = []
        
        row_counts = [0] * rows
        
        for i in range(rows):
            for j in range(cols):
                value = dense_matrix[i][j]
                if value != 0:
                    data.append(value)
                    indices.append(j)
                    row_counts[i] += 1
        
        indptr = [0] * (rows + 1)
        for i in range(rows):
            indptr[i + 1] = indptr[i] + row_counts[i]
        
        return cls(data, indices, indptr, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSRMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix

        m, n = self.shape

        col_counts = [0] * n

        for col_idx in self.indices:
            col_counts[col_idx] += 1

        indptr = [0] * (n + 1)

        for j in range(n):
            indptr[j + 1] = indptr[j] + col_counts[j]

        data = [0] * len(self.data)
        indices = [0] * len(self.indices)

        current_pos = indptr.copy()

        for i in range(m):
            row_start = self.indptr[i]
            row_end = self.indptr[i + 1]

            for k in range(row_start, row_end):
                j = self.indices[k]
                val = self.data[k]

                pos = current_pos[j]

                data[pos] = val
                indices[pos] = i

                current_pos[j] += 1
    
        return CSCMatrix(data, indices, indptr, (m, n))
    
    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSRMatrix в COOMatrix.
        """
        from COO import COOMatrix

        rows, cols = self.shape

        data = []
        row_indices = []
        col_indices = []
        
        for i in range(rows):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            
            # для кажноо элемента в каждой строке находим столбец и значение
            for idx in range(start, end):
                j = self.indices[idx]
                value = self.data[idx]
                
                data.append(value)
                row_indices.append(i)
                col_indices.append(j)
        
        return COOMatrix(data, row_indices, col_indices, self.shape)