from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix
from collections import defaultdict

class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]

        for row in range(rows):
            start = self.indptr[row]
            end = self.indptr[row + 1]
            for idx in range(start, end):
                col = self.indices[idx]
                dense[row][col] = self.data[idx]

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        dense = self.to_dense()
        other_dense = other.to_dense()

        rows, cols = self.shape
        result = [[dense[i][j] + other_dense[i][j] for j in range(cols)] for i in range(rows)]

        return CSRMatrix.from_dense(result)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        if scalar == 0:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)

        new_data = [v * scalar for v in self.data]
        return CSRMatrix(new_data, self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Hint:
        Результат - в CSC формате (с теми же данными, но с интерпретацией столбцов как строк).
        """
        from CSC import CSCMatrix
        return CSCMatrix.from_dense(self.to_dense())

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        from COO import COOMatrix
        if self.shape[1] != other.shape[0]:
            raise ValueError("Несовместимые размеры для умножения")

            # Строим COO-подобное представление другой CSR
        other_rows = defaultdict(list)
        rows_b, cols_b = other.shape
        for row in range(rows_b):
            for idx in range(other.indptr[row], other.indptr[row + 1]):
                col = other.indices[idx]
                val = other.data[idx]
                other_rows[row].append((col, val))

        data, row, col = [], [], []

        for row_a in range(self.shape[0]):
            for idx_a in range(self.indptr[row_a], self.indptr[row_a + 1]):
                col_a = self.indices[idx_a]
                val_a = self.data[idx_a]
                for col_b, val_b in other_rows.get(col_a, []):
                    data.append(val_a * val_b)
                    row.append(row_a)
                    col.append(col_b)

        return COOMatrix(data, row, col, (self.shape[0], other.shape[1]))._to_csr()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0

        data = []
        indices = []
        indptr = [0]

        for i in range(rows):
            count = 0
            for j in range(cols):
                val = dense_matrix[i][j]
                if val != 0:
                    data.append(val)
                    indices.append(j)
                    count += 1
            indptr.append(indptr[-1] + count)

        return cls(data, indices, indptr, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSRMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix
        return CSCMatrix.from_dense(self.to_dense())

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSRMatrix в COOMatrix.
        """
        from COO import COOMatrix
        return COOMatrix.from_dense(self.to_dense())
