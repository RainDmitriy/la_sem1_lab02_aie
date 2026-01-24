from base import Matrix
from types import COOData, COORows, COOCols, Shape, DenseMatrix
from CSR import CSRMatrix
from CSC import CSCMatrix


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.row = row
        self.col = col

        if not (len(data) == len(row) == len(col)):
            raise ValueError("Длины data, row и col должны совпадать")

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]
        # заполняем ненулевыми элементами
        for i in range(len(self.data)):
            r, c, val = self.row[i], self.col[i], self.data[i]
            dense[r][c] = val

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц."""
        # преобразуем обе матрицы в плотные
        dense_self = self.to_dense()
        dense_other = other.to_dense()
        rows, cols = self.shape
        result_dense = [[0.0] * cols for _ in range(rows)]

        for i in range(rows):
            for j in range(cols):
                result_dense[i][j] = dense_self[i][j] + dense_other[i][j]

        return COOMatrix.from_dense(result_dense)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        # умножаем только ненулевые элементы
        new_data = [val * scalar for val in self.data]
        return COOMatrix(new_data, self.row.copy(), self.col.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        # меняем местами строки и столбцы
        return COOMatrix(
            self.data.copy(),
            self.col.copy(),
            self.row.copy(),
            (self.shape[1], self.shape[0])
        )

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""
        # преобразуем в плотную матрицу
        dense_self = self.to_dense()
        dense_other = other.to_dense()
        rows, cols = self.shape[0], other.shape[1]
        inner = self.shape[1]
        result_dense = [[0.0] * cols for _ in range(rows)]

        for i in range(rows):
            for j in range(cols):
                total = 0.0
                for k in range(inner):
                    total += dense_self[i][k] * dense_other[k][j]
                result_dense[i][j] = total

        return COOMatrix.from_dense(result_dense)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        data = []
        row_indices = []
        col_indices = []

        for i in range(rows):
            for j in range(cols):
                val = dense_matrix[i][j]
                if val != 0:
                    data.append(val)
                    row_indices.append(i)
                    col_indices.append(j)

        return cls(data, row_indices, col_indices, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        rows, cols = self.shape
        nnz = len(self.data)

        if nnz == 0:
            return CSCMatrix([], [], [0] * (cols + 1), self.shape)

        col_counts = [0] * cols
        for j in self.col:
            col_counts[j] += 1

        # создаём indptr
        indptr = [0] * (cols + 1)
        for j in range(cols):
            indptr[j + 1] = indptr[j] + col_counts[j]

        # создаём временные массивы для данных
        temp_data = [0.0] * nnz
        temp_row_indices = [0] * nnz
        next_pos = indptr.copy()

        # заполняем данные в порядке столбцов
        for idx in range(nnz):
            col = self.col[idx]
            pos = next_pos[col]
            temp_data[pos] = self.data[idx]
            temp_row_indices[pos] = self.row[idx]
            next_pos[col] += 1

        return CSCMatrix(temp_data, temp_row_indices, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        rows, cols = self.shape
        nnz = len(self.data)

        if nnz == 0:
            return CSRMatrix([], [], [0] * (rows + 1), self.shape)
        row_counts = [0] * rows
        for i in self.row:
            row_counts[i] += 1

        indptr = [0] * (rows + 1)
        for i in range(rows):
            indptr[i + 1] = indptr[i] + row_counts[i]
        temp_data = [0.0] * nnz
        temp_col_indices = [0] * nnz

        next_pos = indptr.copy()
        # заполняем данные в порядке строк
        for idx in range(nnz):
            row = self.row[idx]
            pos = next_pos[row]
            temp_data[pos] = self.data[idx]
            temp_col_indices[pos] = self.col[idx]
            next_pos[row] += 1

        return CSRMatrix(temp_data, temp_col_indices, indptr, self.shape)