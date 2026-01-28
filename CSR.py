from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)

        if len(indptr) != shape[0] + 1:
            raise ValueError("Длина indptr должна быть rows + 1")

        if len(data) != len(indices):
            raise ValueError("Длины data и indices должны совпадать")

        if indptr[-1] != len(data):
            raise ValueError("Последний элемент indptr должен равняться len(data)")

        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0 for _ in range(cols)] for _ in range(rows)]

        for i in range(rows):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for k in range(start, end):
                j = self.indices[k]
                dense[i][j] += self.data[k]   

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        if not isinstance(other, CSRMatrix):
            other = other._to_csr()

        plus_data, plus_indices, plus_indptr = [], [], [0]
        rows, __ = self.shape

        for i in range(rows):
            row_sum = {}

            for k in range(self.indptr[i], self.indptr[i + 1]):
                j = self.indices[k]
                row_sum[j] = row_sum.get(j, 0) + self.data[k]

            for k in range(other.indptr[i], other.indptr[i + 1]):
                j = other.indices[k]
                row_sum[j] = row_sum.get(j, 0) + other.data[k]

            for j in sorted(row_sum):
                v = row_sum[j]
                if v != 0:
                    plus_indices.append(j)
                    plus_data.append(v)

            plus_indptr.append(len(plus_data))

        return CSRMatrix(plus_data, plus_indices, plus_indptr, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        new_data = [v * scalar for v in self.data]
        return CSRMatrix(new_data, list(self.indices), list(self.indptr), self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Hint:
        Результат - в CSC формате (с теми же данными, но с интерпретацией столбцов как строк).
        """
        rows, cols = self.shape
        return CSCMatrix(
            self.data,
            self.indices,
            self.indptr,
            (cols, rows)
        )

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        if self.shape[1] != other.shape[0]:
            raise ValueError("Несовместимые размеры матриц")

        if not isinstance(other, CSRMatrix):
            other = other._to_csr()

        rows, cols = self.shape[0], other.shape[1]
        data, indices, indptr = [], [], [0]

        for i in range(rows):
            row = {}

            for k in range(self.indptr[i], self.indptr[i + 1]):
                j = self.indices[k]
                a = self.data[k]

                for t in range(other.indptr[j], other.indptr[j + 1]):
                    col = other.indices[t]
                    row[col] = row.get(col, 0) + a * other.data[t]

            for j in sorted(row):
                if row[j] != 0:
                    indices.append(j)
                    data.append(row[j])

            indptr.append(len(data))

        return CSRMatrix(data, indices, indptr, (rows, cols))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        data, indices, indptr = [], [], [0]

        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0

        for i in range(rows):
            for j in range(cols):
                v = dense_matrix[i][j]
                if v != 0:
                    data.append(v)
                    indices.append(j)
            indptr.append(len(data))

        return cls(data, indices, indptr, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSRMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix
        rows, cols = self.shape
        nnz = len(self.data)

        col_count = [0] * cols
        for j in self.indices:
            col_count[j] += 1

        indptr = [0] * (cols + 1)
        for i in range(cols):
            indptr[i + 1] = indptr[i] + col_count[i]

        data = [0] * nnz
        indices = [0] * nnz
        current = indptr.copy()

        for i in range(rows):
            for k in range(self.indptr[i], self.indptr[i + 1]):
                j = self.indices[k]
                pos = current[j]
                data[pos] = self.data[k]
                indices[pos] = i
                current[j] += 1

        return CSCMatrix(data, indices, indptr, (rows, cols))
    
    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSRMatrix в COOMatrix.
        """
        from COO import COOMatrix
        rows = []
        num_rows, _ = self.shape

        for i in range(num_rows):
            count = self.indptr[i + 1] - self.indptr[i]
            rows.extend([i] * count)

        return COOMatrix(
            self.data,
            rows,
            self.indices,
            self.shape
        )