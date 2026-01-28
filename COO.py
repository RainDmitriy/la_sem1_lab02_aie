from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.row = row
        self.col = col
        self.data = data

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу."""
        row, col = self.shape
        matrix = [[0.0 for _ in range(col)] for _ in range(row)]
        for row, col, data in zip(self.row, self.col, self.data):
            matrix[row][col] += data

        return matrix

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц."""
        if not isinstance(other, COOMatrix):
            raise TypeError("Складывать можно только COO матрицы")

        rows = self.row + other.row
        cols = self.col + other.col
        values = self.data + other.data

        return COOMatrix(values, rows, cols, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        values = [i * scalar for i in self.data]

        return COOMatrix(values, self.row, self.col, self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        return COOMatrix(self.data,
                         self.col,
                         self.row,
                         (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""
        if self.shape[1] != other.shape[0]:
            print("Умножение матриц невозможно")
            raise ValueError
        csr = self._to_csr()
        if isinstance(other, COOMatrix):
            other_csr = other._to_csr()
            return csr @ other_csr
        res = csr @ other
        return res._to_coo()


    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        rows_num = len(dense_matrix)
        cols_num = len(dense_matrix[0])
        rows = []
        cols = []
        data = []

        for row in range(rows_num):
            for col in range(cols_num):
                value = dense_matrix[row][col]
                if value != 0:
                    rows.append(row)
                    cols.append(col)
                    data.append(value)

        return COOMatrix(data, rows, cols, (rows_num, cols_num))

    def _to_csc(self) -> 'CSCMatrix':
        from CSC import CSCMatrix
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        if not self.data:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)

        sort_coo = sorted(zip(self.row, self.col, self.data),
                                 key=lambda x: (x[1], x[0]))
        rows, cols, data = zip(*sort_coo)
        indptr = [0] * (self.shape[1] + 1)
        numbers = [0] * self.shape[1]
        for c in cols:
            numbers[c] += 1

        cumsum = 0
        for i, num in enumerate(numbers):
            indptr[i] = cumsum
            cumsum += num
        indptr[-1] = cumsum
        return CSCMatrix(data, rows, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        from CSR import CSRMatrix
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        if not self.data:
            return CSRMatrix([], [], [0] * (self.shape[1] + 1), self.shape)

        sort_coo = sorted(zip(self.row, self.col, self.data))
        rows, cols, data = zip(*sort_coo)
        indptr = [0] * (self.shape[0] + 1)
        numbers = [0] * self.shape[0]
        for r in rows:
            numbers[r] += 1

        cum_sum = 0
        for i, num in enumerate(numbers):
            indptr[i] = cum_sum
            cum_sum += num
        indptr[-1] = cum_sum
        return CSRMatrix(data, cols, indptr, self.shape)
