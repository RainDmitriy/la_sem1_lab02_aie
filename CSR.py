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
        dense = [[0.0] * cols for _ in range(rows)]

        for i in range(rows):
            row = dense[i]
            for k in range(self.indptr[i], self.indptr[i + 1]):
                j = self.indices[k]
                row[j] += self.data[k]

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        if not isinstance(other, CSRMatrix):
            other = other._to_csr()

        plus_data, plus_indices, plus_indptr = [], [], [0]
        rows, cols = self.shape

        self_indptr = self.indptr
        self_indices = self.indices
        self_data = self.data
        other_indptr = other.indptr
        other_indices = other.indices
        other_data = other.data

        for i in range(rows):
            row_sum = {}

            for k in range(self_indptr[i], self_indptr[i + 1]):
                j = self_indices[k]
                row_sum[j] = row_sum.get(j, 0) + self_data[k]

            for k in range(other_indptr[i], other_indptr[i + 1]):
                j = other_indices[k]
                row_sum[j] = row_sum.get(j, 0) + other_data[k]

            for j in sorted(row_sum.keys()):
                v = row_sum[j]
                if v != 0:
                    plus_indices.append(j)
                    plus_data.append(v)

            plus_indptr.append(len(plus_data))

        return CSRMatrix(plus_data, plus_indices, plus_indptr, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        new_data = [v * scalar for v in self.data]
        return CSRMatrix(new_data, self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Hint:
        Результат - в CSC формате (с теми же данными, но с интерпретацией столбцов как строк).
        """
        from CSC import CSCMatrix
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
        
        other_indptr = other.indptr
        other_indices = other.indices
        other_data = other.data
        self_indptr = self.indptr
        self_indices = self.indices
        self_data = self.data

        for i in range(rows):
            row = {}

            for k in range(self_indptr[i], self_indptr[i + 1]):
                j = self_indices[k]
                a = self_data[k]

                for t in range(other_indptr[j], other_indptr[j + 1]):
                    col = other_indices[t]
                    row[col] = row.get(col, 0) + a * other_data[t]

            for j in sorted(row.keys()):
                v = row[j]
                if v != 0:
                    indices.append(j)
                    data.append(v)

            indptr.append(len(data))

        return CSRMatrix(data, indices, indptr, (rows, cols))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        data, indices, indptr = [], [], [0]

        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0

        for i in range(rows):
            row_data = dense_matrix[i]
            for j in range(cols):
                v = row_data[j]
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
        indices_data = self.indices
        
        for j in indices_data:
            col_count[j] += 1

        indptr = [0] * (cols + 1)
        for i in range(cols):
            indptr[i + 1] = indptr[i] + col_count[i]

        data = [0.0] * nnz
        indices = [0] * nnz
        current = indptr[:]
        
        self_indptr = self.indptr
        self_data = self.data

        for i in range(rows):
            for k in range(self_indptr[i], self_indptr[i + 1]):
                j = indices_data[k]
                pos = current[j]
                data[pos] = self_data[k]
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