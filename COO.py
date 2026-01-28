from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix

class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        if len(data) != len(row) or len(data) != len(col):
            raise ValueError("Длины data, row и col должны совпадать")

        self.data = data
        self.row = row
        self.col = col

    def iter_nonzero(self):
        """Итератор по ненулевым элементам."""
        for value, i, j in zip(self.data, self.row, self.col):
            yield i, j, value        

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0 for _ in range(cols)] for _ in range(rows)]
        for i, j, value in self.iter_nonzero():
            dense[i][j] += value
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц."""
        if not isinstance(other, COOMatrix):
            other = COOMatrix.from_dense(other.to_dense())

        plus_res = {}

        for i, j, v in self.iter_nonzero():
            plus_res[(i, j)] = plus_res.get((i, j), 0) + v

        for i, j, v in other.iter_nonzero():
            plus_res[(i, j)] = plus_res.get((i, j), 0) + v

        data, row, col = [], [], []
        for (i, j), v in plus_res.items():
            if v != 0:
                data.append(v)
                row.append(i)
                col.append(j)

        return COOMatrix(data, row, col, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        data, row, col = [], [], []

        for i, j, v in self.iter_nonzero():
            new_v = v * scalar
            if abs(new_v) > 1e-10:
                data.append(new_v)
                row.append(i)
                col.append(j)

        return COOMatrix(data, row, col, self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        data, row, col = [], [], []

        for i, j, v in self.iter_nonzero():
            data.append(v)
            row.append(j)
            col.append(i)

        rows, cols = self.shape
        new_shape = (cols, rows)

        return COOMatrix(data, row, col, new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""
        if not isinstance(other, COOMatrix):
            other = COOMatrix.from_dense(other.to_dense())

        first_rows, first_cols = self.shape
        second_rows, second_cols = other.shape

        mult_res = {}

        for i1, j1, v1 in self.iter_nonzero():
            for i2, j2, v2 in other.iter_nonzero():
                if j1 == i2:
                    key = (i1, j2)
                    mult_res[key] = mult_res.get(key, 0) + v1 * v2

        data, row, col = [], [], []
        for (i, j), v in mult_res.items():
            if v != 0:
                data.append(v)
                row.append(i)
                col.append(j)

        new_shape = (first_rows, second_cols)
        return COOMatrix(data, row, col, new_shape)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        data, row, col = [], [], []

        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0

        for i in range(rows):
            for j in range(cols):
                value = dense_matrix[i][j]
                if value != 0:
                    data.append(value)
                    row.append(i)
                    col.append(j)

        return cls(data, row, col, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix
        _, num_cols = self.shape

        cols_elements = [[] for _ in range(num_cols)]
        for v, r, c in zip(self.data, self.row, self.col):
            cols_elements[c].append((r, v))

        data, row, indptr = [], [], [0]

        for c in range(num_cols):
            for r, v in cols_elements[c]:
                data.append(v)
                row.append(r)
            indptr.append(len(data))

        return CSCMatrix(data, row, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix
        num_rows, _ = self.shape

        rows_elements = [[] for _ in range(num_rows)]
        for v, r, c in zip(self.data, self.row, self.col):
            rows_elements[r].append((c, v))

        data, indices, indptr = [], [], [0]

        for i in range(num_rows):
            row_items = sorted(rows_elements[i])
            for c, v in rows_elements[i]:
                data.append(v)
                indices.append(c)
            indptr.append(len(data))

        return CSRMatrix(data, indices, indptr, self.shape)
