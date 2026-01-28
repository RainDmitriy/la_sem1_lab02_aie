from base import Matrix
from collections import defaultdict
from type import COOData, COORows, COOCols, Shape, DenseMatrix

class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)

        self.data = data
        self.row = row
        self.col = col

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]

        for v, i, j in zip(self.data, self.row, self.col):
            dense[i][j] = v

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        result = defaultdict(float)
        for v, r, c in zip(self.data, self.row, self.col):
            result[(r, c)] += v
        for v, r, c in zip(other.data, other.row, other.col):
            result[(r, c)] += v
        data, row, col = [], [], []
        for (r, c), v in result.items():
            if v != 0:
                data.append(v)
                row.append(r)
                col.append(c)
        return COOMatrix(data, row, col, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        if scalar == 0:
            return COOMatrix([], [], [], self.shape)
        new_data = [v * scalar for v in self.data]
        return COOMatrix(new_data, self.row[:], self.col[:], self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        r, c = self.shape
        return COOMatrix(
            self.data[:],
            self.col[:],
            self.row[:],
            (c, r)
        )

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""
        if self.shape[1] != other.shape[0]:
            raise ValueError("Несовместимые размеры для умножения")

        other_row_dict = defaultdict(list)
        for v, r, c in zip(other.data, other.row, other.col):
            other_row_dict[r].append((c, v))

        result = defaultdict(float)

        for v1, r1, c1 in zip(self.data, self.row, self.col):
            for c2, v2 in other_row_dict.get(c1, []):
                result[(r1, c2)] += v1 * v2

        data, row, col = [], [], []
        for (r, c), v in result.items():
            if v != 0:
                data.append(v)
                row.append(r)
                col.append(c)

        shape = (self.shape[0], other.shape[1])
        return COOMatrix(data, row, col, shape)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows else 0

        data, row, col = [], [], []

        for i in range(rows):
            for j in range(cols):
                if dense_matrix[i][j] != 0:
                    data.append(dense_matrix[i][j])
                    row.append(i)
                    col.append(j)

        return cls(data, row, col, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix
        elements = sorted(zip(self.col, self.row, self.data))
        sorted_col = [e[0] for e in elements]
        sorted_row = [e[1] for e in elements]
        sorted_data = [e[2] for e in elements]
        cols = self.shape[1]
        indptr = [0] * (cols + 1)
        for c in sorted_col:
            indptr[c + 1] += 1
        for i in range(1, len(indptr)):
            indptr[i] += indptr[i - 1]
        return CSCMatrix(sorted_data, sorted_row, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix
        elements = sorted(zip(self.row, self.col, self.data))
        sorted_row = [e[0] for e in elements]
        sorted_col = [e[1] for e in elements]
        sorted_data = [e[2] for e in elements]
        rows = self.shape[0]
        indptr = [0] * (rows + 1)
        for r in sorted_row:
            indptr[r + 1] += 1
        for i in range(1, len(indptr)):
            indptr[i] += indptr[i - 1]
        return CSRMatrix(sorted_data, sorted_col, indptr, self.shape)