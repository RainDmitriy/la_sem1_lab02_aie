from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.row = row
        self.col = col
        self.shape = shape

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу."""
        n, m = self.shape
        dense_matrix = [[0] * m for _ in range(n)]

        k = len(self.row)
        for i in range(k):
            col, row, val = self.col[i], self.row[i], self.data[i]
            dense_matrix[row][col] = val

        return dense_matrix

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц."""
        all_row = self.row + other.row
        all_col = self.col + other.col
        all_val = self.data + other.data

        merged_coords = dict()
        for r, c, v in zip(all_row, all_col, all_val):
            key = (r, c)
            merged_coords[key] = merged_coords.get(key, 0) + v

        sum_row, sum_col, sum_val = list(), list(), list()
        for (row, col), val in sorted(merged_coords.items()):
            if val != 0:
                sum_row.append(row)
                sum_col.append(col)
                sum_val.append(val)

        return COOMatrix(sum_val, sum_row, sum_col, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        if scalar == 0:
            return COOMatrix(list(), list(), list(), self.shape)

        new_data = [x * scalar for x in self.data]

        return COOMatrix(new_data, self.row, self.col, self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        data = list()
        for r, c, v in zip(self.row, self.col, self.data):
            data.append([c, r, v])

        data.sort(key=lambda x: x[0])

        new_row, new_col, new_val = list(), list(), list()
        new_shape = (self.shape[1], self.shape[0])
        for r, c, v in data:
            new_row.append(r)
            new_col.append(c)
            new_val.append(v)

        return COOMatrix(new_val, new_row, new_col, new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""
        n = self.shape[0]
        k = other.shape[1]
        new_row, new_col, new_val = list(), list(), list()
        new_shape = (n, k)

        merged_coords = dict()
        for r_1, c_1, v_1 in zip(self.row, self.col, self.data):
            for r_2, c_2, v_2 in zip(other.row, other.col, other.data):
                if c_1 == r_2:
                    key = (r_1, c_2)
                    merged_coords[key] = merged_coords.get(key, 0) + v_1 * v_2

        for (r, c), v in merged_coords.items():
            if v != 0:
                new_row.append(r)
                new_col.append(c)
                new_val.append(v)

        return COOMatrix(new_val, new_row, new_col, new_shape)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        n, m = len(dense_matrix), len(dense_matrix[0])
        shape = (n, m)
        rows, cols, val = list(), list(), list()

        for i in range(n):
            for j in range(m):
                if dense_matrix[i][j] != 0:
                    rows.append(i)
                    cols.append(j)
                    val.append(dense_matrix[i][j])

        return COOMatrix(val, rows, cols, shape)

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix

        all_to_sort = list(zip(self.row, self.col, self.data))
        all_to_sort.sort(key=lambda x: x[1])

        if all_to_sort:
            sorted_rows, sorted_cols, sorted_data = zip(*all_to_sort)
        else:
            sorted_rows, sorted_cols, sorted_data = [], [], []

        data = list(sorted_data)
        indices = list(sorted_rows)
        n = self.shape[1]
        indptr = [0] * (n + 1)

        for i in sorted_cols:
            indptr[i + 1] += 1

        for i in range(n):
            indptr[i + 1] += indptr[i]

        return CSCMatrix(data, indices, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix

        all_to_sort = list(zip(self.row, self.col, self.data))
        all_to_sort.sort()

        if all_to_sort:
            sorted_rows, sorted_cols, sorted_data = zip(*all_to_sort)
        else:
            sorted_rows, sorted_cols, sorted_data = [], [], []

        data = list(sorted_data)
        indices = list(sorted_cols)
        n = self.shape[0]
        indptr = [0] * (n + 1)

        for i in sorted_rows:
            indptr[i + 1] += 1

        for i in range(n):
            indptr[i + 1] += indptr[i]

        return CSRMatrix(data, indices, indptr, self.shape)
