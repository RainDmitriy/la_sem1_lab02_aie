from base import Matrix
from matrix_types import COOData, COORows, COOCols, Shape, DenseMatrix

class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.row = row
        self.col = col
        self.nnz = len(data) #колво ненулевых элементов

        if len(data) != len(row) or len(data) != len(col):
            raise ValueError("data, row и col не совпадают")
        for r, c in zip(row, col):
            if r < 0 or r >= shape[0] or c < 0 or c >= shape[1]:
                raise ValueError("индекс за границой матрицы")

    def to_dense(self) -> DenseMatrix:
        """из COO в плотную матрицу"""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]
        for value, r, c in zip(self.data, self.row, self.col):
            dense[r][c] = value
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """сложение матриц."""
        dense_self = self.to_dense()
        dense_other = other.to_dense()
        result_dense = []
        for i in range(self.shape[0]):
            row = []
            for j in range(self.shape[1]):
                row.append(dense_self[i][j] + dense_other[i][j])
            result_dense.append(row)

        return COOMatrix.from_dense(result_dense)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """умножение на скаляр"""
        new_data = [value * scalar for value in self.data]
        return COOMatrix(new_data, self.row.copy(), self.col.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """транспонирование"""
        new_shape = (self.shape[1], self.shape[0])
        return COOMatrix(self.data.copy(), self.col.copy(), self.row.copy(), new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """умножение матриц"""
        if self.shape[1] != other.shape[0]:
            raise ValueError("несовместимые размерности")
        dense_self = self.to_dense()
        dense_other = other.to_dense()
        result_rows = self.shape[0]
        result_cols = other.shape[1]
        result_dense = [[0.0] * result_cols for _ in range(result_rows)]
        for i in range(result_rows):
            for j in range(result_cols):
                total = 0.0
                for k in range(self.shape[1]):
                    total += dense_self[i][k] * dense_other[k][j]
                result_dense[i][j] = total

        return COOMatrix.from_dense(result_dense)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """создание coo из плотной матрицы"""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        data = []
        row_indices = []
        col_indices = []
        for i in range(rows):
            for j in range(cols):
                value = dense_matrix[i][j]
                if abs(value) > 1e-10:  #ноль = очень маленькие значения
                    data.append(value)
                    row_indices.append(i)
                    col_indices.append(j)

        return cls(data, row_indices, col_indices, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """преобразование COOMatrix в CSCMatrix"""
        from CSC import CSCMatrix
        #сортируем элементы по столбцам и строкам
        sorted_indices = sorted(range(self.nnz), key=lambda k: (self.col[k], self.row[k]))
        data = []
        indices = []
        indptr = [0] * (self.shape[1] + 1)
        current_col = 0
        for idx in sorted_indices:
            col = self.col[idx]
            while current_col < col:
                indptr[current_col + 1] = len(data)
                current_col += 1
            data.append(self.data[idx])
            indices.append(self.row[idx])
        for col in range(current_col, self.shape[1]):
            indptr[col + 1] = len(data)

        return CSCMatrix(data, indices, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """преобразование COOMatrix в CSRMatrix"""
        from CSR import CSRMatrix
        sorted_indices = sorted(range(self.nnz), key=lambda k: (self.row[k], self.col[k]))
        data = []
        indices = []
        indptr = [0] * (self.shape[0] + 1)
        current_row = 0
        for idx in sorted_indices:
            row = self.row[idx]
            while current_row < row:
                indptr[current_row + 1] = len(data)
                current_row += 1
            data.append(self.data[idx])
            indices.append(self.col[idx])
        for row in range(current_row, self.shape[0]):
            indptr[row + 1] = len(data)

        return CSRMatrix(data, indices, indptr, self.shape)