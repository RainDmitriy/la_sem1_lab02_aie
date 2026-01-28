from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.row = row
        self.col = col
        self.nnz = len(data)

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу."""
        n_rows, n_cols = self.shape
        dense = [[0.0] * n_cols for _ in range(n_rows)]
        for val, i, j in zip(self.data, self.row, self.col):
            dense[i][j] = val
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц."""
        other_coo = other if isinstance(other, COOMatrix) else other._to_coo()
        if self.shape != other_coo.shape:
            raise ValueError()
        sum_dict: dict[tuple[int, int], float] = {}
        for v, i, j in zip(self.data, self.row, self.col):
            sum_dict[(i, j)] = sum_dict.get((i, j), 0.0) + v
        for v, i, j in zip(other_coo.data, other_coo.row, other_coo.col):
            sum_dict[(i, j)] = sum_dict.get((i, j), 0.0) + v
        new_data, new_row, new_col = [], [], []
        for (i, j), v in sorted(sum_dict.items()):
            if abs(v) > 1e-12:
                new_data.append(v)
                new_row.append(i)
                new_col.append(j)
        return COOMatrix(new_data, new_row, new_col, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        new_data = [val * scalar for val in self.data]
        return COOMatrix(new_data, self.row.copy(), self.col.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        n_rows, n_cols = self.shape
        return COOMatrix(
            self.data.copy(),
            self.col.copy(),
            self.row.copy(),
            (n_cols, n_rows)
        )

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""
        n_rows, n_cols = self.shape
        other_n_rows, other_n_cols = other.shape
        if n_cols != other_n_rows:
            raise ValueError()
        if not isinstance(other, COOMatrix):
            other_coo = other._to_coo()
        else:
            other_coo = other
        other_rows_dict = {}
        for idx in range(other_coo.nnz):
            i = other_coo.row[idx]
            if i not in other_rows_dict:
                other_rows_dict[i] = []
            other_rows_dict[i].append((other_coo.col[idx], other_coo.data[idx]))
        result_dict = {}
        for i in range(n_rows):
            row_elements = []
            for idx in range(self.nnz):
                if self.row[idx] == i:
                    row_elements.append((self.col[idx], self.data[idx]))
            for col_self, val_self in row_elements:
                if col_self in other_rows_dict:
                    for col_other, val_other in other_rows_dict[col_self]:
                        key = (i, col_other)
                        result_dict[key] = result_dict.get(key, 0.0) + val_self * val_other
        new_data = []
        new_row = []
        new_col = []
        for (i, j), val in result_dict.items():
            if abs(val) > 1e-12:
                new_data.append(val)
                new_row.append(i)
                new_col.append(j)
        return COOMatrix(new_data, new_row, new_col, (n_rows, other_n_cols))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        n_rows = len(dense_matrix)
        n_cols = len(dense_matrix[0]) if n_rows > 0 else 0
        data = []
        row = []
        col = []
        for i in range(n_rows):
            for j in range(n_cols):
                val = dense_matrix[i][j]
                if abs(val) > 1e-12:
                    data.append(val)
                    row.append(i)
                    col.append(j)
        return cls(data, row, col, (n_rows, n_cols))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix
        n_rows, n_cols = self.shape
        sorted_idx = sorted(range(self.nnz), key=lambda k: (self.col[k], self.row[k]))
        data = [self.data[i] for i in sorted_idx]
        indices = [self.row[i] for i in sorted_idx]
        indptr = [0] * (n_cols + 1)
        for i in sorted_idx:
            col = self.col[i]
            indptr[col + 1] += 1
        for j in range(n_cols):
            indptr[j + 1] += indptr[j]
        return CSCMatrix(data, indices, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix
        n_rows, n_cols = self.shape
        sorted_idx = sorted(range(self.nnz), key=lambda k: (self.row[k], self.col[k]))
        data = [self.data[i] for i in sorted_idx]
        indices = [self.col[i] for i in sorted_idx]
        indptr = [0] * (n_rows + 1)
        for i in sorted_idx:
            row = self.row[i]
            indptr[row + 1] += 1
        for i in range(n_rows):
            indptr[i + 1] += indptr[i]
        return CSRMatrix(data, indices, indptr, self.shape)

    def _to_coo(self) -> 'COOMatrix':
        return self
