from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix
from collections import defaultdict


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
        for i in range(len(self.data)):
            r, c, val = self.row[i], self.col[i], self.data[i]
            dense[r][c] = val
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц напрямую в разреженном формате."""
        if not isinstance(other, COOMatrix):
            other_coo = other._to_coo() if hasattr(other, '_to_coo') else COOMatrix.from_dense(other.to_dense())
        else:
            other_coo = other

        rows, cols = self.shape

        result_dict = defaultdict(float)

        for idx in range(len(self.data)):
            key = (self.row[idx], self.col[idx])
            result_dict[key] += self.data[idx]

        for idx in range(len(other_coo.data)):
            key = (other_coo.row[idx], other_coo.col[idx])
            result_dict[key] += other_coo.data[idx]

        data = []
        row_indices = []
        col_indices = []

        for (r, c), val in result_dict.items():
            if abs(val) > 1e-12:
                data.append(val)
                row_indices.append(r)
                col_indices.append(c)

        return COOMatrix(data, row_indices, col_indices, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        new_data = [val * scalar for val in self.data]
        return COOMatrix(new_data, self.row.copy(), self.col.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        return COOMatrix(
            self.data.copy(),
            self.col.copy(),
            self.row.copy(),
            (self.shape[1], self.shape[0])
        )

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение матриц напрямую в разреженном формате."""
        # Для умножения преобразуем в CSR (оптимальнее для умножения)
        csr_self = self._to_csr()
        return csr_self._matmul_impl(other)

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
                if abs(val) > 1e-12:  # Порог для нуля
                    data.append(val)
                    row_indices.append(i)
                    col_indices.append(j)

        return cls(data, row_indices, col_indices, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix

        rows, cols = self.shape
        nnz = len(self.data)

        if nnz == 0:
            return CSCMatrix([], [], [0] * (cols + 1), self.shape)

        col_counts = [0] * cols
        for j in self.col:
            col_counts[j] += 1
        indptr = [0] * (cols + 1)
        for j in range(cols):
            indptr[j + 1] = indptr[j] + col_counts[j]

        temp_data = [0.0] * nnz
        temp_indices = [0] * nnz
        next_pos = indptr.copy()

        sorted_indices = sorted(range(nnz), key=lambda i: (self.col[i], self.row[i]))

        for idx in sorted_indices:
            col = self.col[idx]
            pos = next_pos[col]
            temp_data[pos] = self.data[idx]
            temp_indices[pos] = self.row[idx]
            next_pos[col] += 1

        return CSCMatrix(temp_data, temp_indices, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix

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
        temp_indices = [0] * nnz
        next_pos = indptr.copy()
        sorted_indices = sorted(range(nnz), key=lambda i: (self.row[i], self.col[i]))

        for idx in sorted_indices:
            row = self.row[idx]
            pos = next_pos[row]
            temp_data[pos] = self.data[idx]
            temp_indices[pos] = self.col[idx]
            next_pos[row] += 1

        return CSRMatrix(temp_data, temp_indices, indptr, self.shape)

    @classmethod
    def from_coo(cls, data: COOData, rows: COORows, cols: COOCols, shape: Shape) -> 'COOMatrix':
        """Создание COO из COO данных."""
        return cls(data, rows, cols, shape)