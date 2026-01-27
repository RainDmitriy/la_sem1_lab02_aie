from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix
from CSC import CSCMatrix
from CSR import CSRMatrix


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
        if isinstance(other, COOMatrix):
            other_coo = other
        else:
            other_coo = other._to_coo()
        result_dict = {}
        for idx in range(self.nnz):
            key = (self.row[idx], self.col[idx])
            result_dict[key] = self.data[idx]
        for idx in range(other_coo.nnz):
            key = (other_coo.row[idx], other_coo.col[idx])
            if key in result_dict:
                result_dict[key] += other_coo.data[idx]
            else:
                result_dict[key] = other_coo.data[idx]
        new_data = []
        new_row = []
        new_col = []
        for (i, j), val in result_dict.items():
            if abs(val) > 1e-12:
                new_data.append(val)
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
        n_rows, n_cols = self.shape
        sorted_indices = sorted(range(self.nnz), key=lambda idx: (self.col[idx], self.row[idx]))
        data = []
        indices = []
        indptr = [0] * (n_cols + 1)
        current_col = -1
        for idx in sorted_indices:
            col_idx = self.col[idx]
            while current_col < col_idx:
                current_col += 1
                indptr[current_col + 1] = indptr[current_col]
            data.append(self.data[idx])
            indices.append(self.row[idx])
            indptr[col_idx + 1] += 1
        while current_col < n_cols - 1:
            current_col += 1
            indptr[current_col + 1] = indptr[current_col]
        return CSCMatrix(data, indices, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        n_rows, n_cols = self.shape
        sorted_indices = sorted(range(self.nnz), key=lambda idx: (self.row[idx], self.col[idx]))
        data = []
        indices = []
        indptr = [0] * (n_rows + 1)
        current_row = -1
        for idx in sorted_indices:
            row_idx = self.row[idx]
            while current_row < row_idx:
                current_row += 1
                indptr[current_row + 1] = indptr[current_row]
            data.append(self.data[idx])
            indices.append(self.col[idx])
            indptr[row_idx + 1] += 1
        while current_row < n_rows - 1:
            current_row += 1
            indptr[current_row + 1] = indptr[current_row]
        return CSRMatrix(data, indices, indptr, self.shape)

