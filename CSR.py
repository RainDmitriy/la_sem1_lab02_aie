

from typing import List, TYPE_CHECKING

from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix

if TYPE_CHECKING:
    from COO import COOMatrix
    from CSC import CSCMatrix


class CSRMatrix(Matrix):

    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape) -> None:
        super().__init__(shape)
        rows, _ = shape
        if len(indptr) != rows + 1:
            raise ValueError("Размер indptr должен быть на 1 больше количества строк")
        if indptr[0] != 0:
            raise ValueError("Первый элемент indptr должен быть 0")
        if indptr[-1] != len(data):
            raise ValueError("Последний элемент indptr должен равняться числу ненулевых элементов")
        if len(data) != len(indices):
            raise ValueError("Длина data и indices должна совпадать")
        self.data = list(data)
        self.indices = list(indices)
        self.indptr = list(indptr)

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        m, n = self.shape
        dense = [[0.0] * n for _ in range(m)]
        for i in range(m):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for p in range(start, end):
                j = self.indices[p]
                dense[i][j] = self.data[p]
        return dense

    def _add_impl(self, other: Matrix) -> Matrix:
        """Сложение CSR матриц."""
        if not isinstance(other, CSRMatrix):
            other = other._to_csr()
        m, _ = self.shape
        result_data: CSRData = []
        result_indices: CSRIndices = []
        result_indptr: CSRIndptr = [0] * (m + 1)
        
        for i in range(m):
            a_start = self.indptr[i]
            a_end = self.indptr[i + 1]
            b_start = other.indptr[i]
            b_end = other.indptr[i + 1]
            pa = a_start
            pb = b_start
            
            while pa < a_end and pb < b_end:
                col_a = self.indices[pa]
                col_b = other.indices[pb]
                if col_a == col_b:
                    val = self.data[pa] + other.data[pb]
                    if abs(val) > 1e-14:
                        result_indices.append(col_a)
                        result_data.append(val)
                    pa += 1
                    pb += 1
                elif col_a < col_b:
                    val = self.data[pa]
                    if abs(val) > 1e-14:
                        result_indices.append(col_a)
                        result_data.append(val)
                    pa += 1
                else:
                    val = other.data[pb]
                    if abs(val) > 1e-14:
                        result_indices.append(col_b)
                        result_data.append(val)
                    pb += 1
            
            while pa < a_end:
                col_a = self.indices[pa]
                val = self.data[pa]
                if abs(val) > 1e-14:
                    result_indices.append(col_a)
                    result_data.append(val)
                pa += 1
            
            while pb < b_end:
                col_b = other.indices[pb]
                val = other.data[pb]
                if abs(val) > 1e-14:
                    result_indices.append(col_b)
                    result_data.append(val)
                pb += 1
            result_indptr[i + 1] = len(result_data)
        return CSRMatrix(result_data, result_indices, result_indptr, self.shape)

    def _mul_impl(self, scalar: float) -> Matrix:
        """Умножение CSR на скаляр."""
        if scalar == 0:
            m, _ = self.shape
            return CSRMatrix([], [], [0] * (m + 1), self.shape)
        new_data = [v * scalar for v in self.data]
        return CSRMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> Matrix:
        """
        Транспонирование CSR матрицы.
        Hint:
        Результат - в CSC формате (с теми же данными, но с интерпретацией столбцов как строк).
        """
        from CSC import CSCMatrix
        m, n = self.shape
        new_rows, new_cols = n, m
        col_counts: list[int] = [0] * new_cols
        for i in range(m):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            col_counts[i] = end - start
        new_indptr: CSRIndptr = [0] * (new_cols + 1)
        for j in range(new_cols):
            new_indptr[j + 1] = new_indptr[j] + col_counts[j]
        new_data: CSRData = [0.0] * len(self.data)
        new_indices: CSRIndices = [0] * len(self.indices)
        col_positions = new_indptr.copy()
        for i in range(m):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for idx in range(start, end):
                j = self.indices[idx]
                val = self.data[idx]
                pos = col_positions[i]
                new_data[pos] = val
                new_indices[pos] = j
                col_positions[i] += 1
        return CSCMatrix(new_data, new_indices, new_indptr, (new_rows, new_cols))

    def _matmul_impl(self, other: Matrix) -> Matrix:
        """Умножение CSR матриц."""
        if not isinstance(other, CSRMatrix):
            other = other._to_csr()
        m, _ = self.shape
        _, n = other.shape
        result_data: CSRData = []
        result_indices: CSRIndices = []
        result_indptr: CSRIndptr = [0] * (m + 1)
        
        for i in range(m):
            row_dict: dict[int, float] = {}
            a_start = self.indptr[i]
            a_end = self.indptr[i + 1]
            
            for pa in range(a_start, a_end):
                k = self.indices[pa]
                a_val = self.data[pa]
                
                b_start = other.indptr[k]
                b_end = other.indptr[k + 1]
                for pb in range(b_start, b_end):
                    j = other.indices[pb]
                    b_val = other.data[pb]
                    row_dict[j] = row_dict.get(j, 0.0) + a_val * b_val
            
            for j in sorted(row_dict.keys()):
                v = row_dict[j]
                if abs(v) > 1e-14:
                    result_indices.append(j)
                    result_data.append(v)
            result_indptr[i + 1] = len(result_data)
        return CSRMatrix(result_data, result_indices, result_indptr, (m, n))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> "CSRMatrix":
        """Создание CSR из плотной матрицы."""
        if not dense_matrix or not dense_matrix[0]:
            return cls([], [], [0, 0], (0, 0))
        m = len(dense_matrix)
        n = len(dense_matrix[0])
        data: CSRData = []
        indices: CSRIndices = []
        indptr: CSRIndptr = [0] * (m + 1)
        for i in range(m):
            count = 0
            for j, value in enumerate(dense_matrix[i]):
                if value != 0:
                    data.append(value)
                    indices.append(j)
                    count += 1
            indptr[i + 1] = indptr[i] + count
        return cls(data, indices, indptr, (m, n))

    def _to_csc(self) -> "CSCMatrix":
        """
        Преобразование CSRMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix
        m, n = self.shape
        col_counts: List[int] = [0] * n
        for i in range(m):
            for p in range(self.indptr[i], self.indptr[i + 1]):
                col = self.indices[p]
                col_counts[col] += 1
        indptr: CSRIndptr = [0] * (n + 1)
        for j in range(n):
            indptr[j + 1] = indptr[j] + col_counts[j]
        
        data: List[float] = [0.0] * len(self.data)
        indices: List[int] = [0] * len(self.indices)
        current_pos = indptr.copy()
        
        for i in range(m):
            row_start = self.indptr[i]
            row_end = self.indptr[i + 1]
            for p in range(row_start, row_end):
                col = self.indices[p]
                val = self.data[p]
                pos = current_pos[col]
                data[pos] = val
                indices[pos] = i
                current_pos[col] += 1
        return CSCMatrix(data, indices, indptr, (m, n))

    def _to_coo(self) -> "COOMatrix":
        """
        Преобразование CSRMatrix в COOMatrix.
        """
        from COO import COOMatrix
        m, n = self.shape
        data_list: List[float] = []
        row_list: List[int] = []
        col_list: List[int] = []
        for i in range(m):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for p in range(start, end):
                data_list.append(self.data[p])
                row_list.append(i)
                col_list.append(self.indices[p])
        return COOMatrix(data_list, row_list, col_list, (m, n))