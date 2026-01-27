from base import Matrix
from matrix_types import *

class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.row = row
        self.col = col

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу."""
        n, m = self.shape
        result = [[0 for _ in range(m)] for _ in range(n)]

        for i in range(len(self.data)):
            result[self.row[i]][self.col[i]] = self.data[i]

        return result

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц."""
        if self.shape != other.shape:
            raise ValueError("Размерности матриц не совпадают")
        
        n, m = self.shape
        result = [[0 for _ in range(m)] for _ in range(n)]

        for i in range(len(self.data)):
            result[self.row[i]][self.col[i]] += self.data[i]

        for i in range(len(other.data)):
            result[other.row[i]][other.col[i]] += other.data[i]

        new_row, new_col, new_data = [], [], []
        for i in range(n):
            for j in range(m):
                if result[i][j] != 0:
                    new_row.append(i)
                    new_col.append(j)
                    new_data.append(result[i][j])

        return COOMatrix(new_data, new_row, new_col, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        new_data = [self.data[i] * scalar for i in range(len(self.data))]
        return COOMatrix(new_data, self.row, self.col, self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        return COOMatrix(
            data=self.data.copy(),
            row=self.col.copy(),
            col=self.row.copy(),
            shape=(self.shape[1], self.shape[0])
        )

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""
        if self.shape[1] != other.shape[0]:
            raise ValueError("Несовместимые размерности для умножения")
        
        result = {}  # {(row, col): value}

        for i in range(len(self.data)):
            r1, c1, v1 = self.row[i], self.col[i], self.data[i]
            for j in range(len(other.data)):
                r2, c2, v2 = other.row[j], other.col[j], other.data[j]
                if c1 == r2:
                    if (r1, c2) in result:
                        result[(r1, c2)] += v1 * v2
                    else:
                        result[(r1, c2)] = v1 * v2

        new_row = []
        new_col = []
        new_data = []
        for (r, c), v in result.items():
            if v != 0:
                new_row.append(r)
                new_col.append(c)
                new_data.append(v)

        return COOMatrix(new_data, new_row, new_col, (self.shape[0], other.shape[1]))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        row, col, data = [], [], []

        for i in range(len(dense_matrix)):
            for j in range(len(dense_matrix[i])):
                if dense_matrix[i][j] != 0:
                    row.append(i)
                    col.append(j)
                    data.append(dense_matrix[i][j])

        return COOMatrix(data, row, col, (len(dense_matrix), len(dense_matrix[0])))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix

        _, n_cols = self.shape
        nnz = len(self.data)

        idx = sorted(range(nnz), key=lambda i: (self.col[i], self.row[i]))

        data = [self.data[i] for i in idx]
        indices = [self.row[i] for i in idx]
        indptr = [0] * (n_cols + 1)

        for i in range(nnz):
            indptr[self.col[idx[i]] + 1] += 1

        for i in range(1, n_cols + 1):
            indptr[i] += indptr[i - 1]

        return CSCMatrix(
            data=data,
            indices=indices,
            indptr=indptr,
            shape=self.shape
        )

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix

        n_rows, _ = self.shape
        nnz = len(self.data)

        data = [0] * nnz
        indices = [0] * nnz
        indptr = [0] * (n_rows + 1)

        idx = sorted(range(nnz), key=lambda i: (self.row[i], self.col[i]))

        data = [self.data[i] for i in idx]
        indices = [self.col[i] for i in idx]

        for i in range(nnz):
            indptr[self.row[idx[i]] + 1] += 1

        for i in range(1, n_rows + 1):
            indptr[i] += indptr[i - 1]

        return CSRMatrix(
            data=data,
            indices=indices,
            indptr=indptr,
            shape=self.shape
        )
