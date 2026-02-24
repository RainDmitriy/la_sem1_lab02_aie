from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        if len(indptr) != shape[0] + 1:
            raise ValueError("CSR: indptr должен быть длины rows + 1")
        if len(data) != len(indices):
            raise ValueError("CSR: data и indices должны быть одной длины")
        self.data = list(data)
        self.indices = list(indices)
        self.indptr = list(indptr) 

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0 for _ in range(cols)] for _ in range(rows)]
        for row in range(rows):
            start = self.indptr[row]
            end = self.indptr[row + 1]
            for idx in range(start, end):
                col = self.indices[idx]
                dense[row][col] = self.data[idx]
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        if not isinstance(other, CSRMatrix):
            other = other._to_csr()
        if self.shape != other.shape:
            raise ValueError("Shapes must match")
        n, _ = self.shape
        result_data = []
        result_indices = []
        result_indptr = [0]

        for i in range(n):
            row_vals = {}

            for idx in range(self.indptr[i], self.indptr[i+1]):
                col = self.indices[idx]
                row_vals[col] = self.data[idx]

            for idx in range(other.indptr[i], other.indptr[i+1]):
                col = other.indices[idx]
                row_vals[col] = row_vals.get(col, 0) + other.data[idx]

            for col, val in row_vals.items():
                val = row_vals[col]
                if val != 0:
                    result_data.append(val)
                    result_indices.append(col)

            result_indptr.append(len(result_data))

        return CSRMatrix(result_data, result_indices, result_indptr, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        rows, _ = self.shape
        if scalar == 0:
            return CSRMatrix([], [], [0] * (rows + 1), self.shape)
        new_data = [val * scalar for val in self.data]
        return CSRMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Hint:
        Результат - в CSC формате (с теми же данными, но с интерпретацией столбцов как строк).
        """
        from CSC import CSCMatrix
        rows, cols = self.shape
        nnz = len(self.data)
        col_counts = [0] * cols
        for col in self.indices:
            col_counts[col] += 1
        col_ptr = [0] * (cols + 1)
        for i in range(cols):
            col_ptr[i + 1] = col_ptr[i] + col_counts[i]
        data = [0] * nnz
        row_ind = [0] * nnz
        counter = col_ptr.copy()
        for row in range(rows):
            start = self.indptr[row]
            end = self.indptr[row + 1]
            for idx in range(start, end):
                col = self.indices[idx]
                pos = counter[col]
                data[pos] = self.data[idx]
                row_ind[pos] = row
                counter[col] += 1
        csc = CSCMatrix(data, row_ind, col_ptr, (cols, rows))
        return csc._to_csr()

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        if not isinstance(other, CSRMatrix):
            other = other._to_csr()
        n, m = self.shape
        m2, p = other.shape
        if m != m2:
            raise ValueError("Shapes not aligned for matmul")

        result_data = []
        result_indices = []
        result_indptr = [0]

        for i in range(n):
            row_result = {}
            start_a = self.indptr[i]
            end_a = self.indptr[i + 1]

            for idx_a in range(start_a, end_a):
                k = self.indices[idx_a]
                val_a = self.data[idx_a]
                start_b = other.indptr[k]
                end_b = other.indptr[k + 1]

                for idx_b in range(start_b, end_b):
                    j = other.indices[idx_b]
                    val_b = other.data[idx_b]
                    row_result[j] = row_result.get(j, 0) + val_a * val_b

            for j, val in row_result.items():
                if val != 0:
                    result_data.append(val)
                    result_indices.append(j)
            result_indptr.append(len(result_data))

        return CSRMatrix(result_data, result_indices, result_indptr, (n, p))


    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        data = []
        indices = []
        indptr = [0]
        for i in range(rows):
            for j in range(cols):
                if dense_matrix[i][j] != 0:
                    data.append(dense_matrix[i][j])
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
        col_counts = [0] * cols
        for col in self.indices:
            col_counts[col] += 1
        col_ptr = [0] * (cols + 1)
        for col in range(cols):
            col_ptr[col + 1] = col_ptr[col] + col_counts[col]
        data = [0] * nnz
        row_ind = [0] * nnz
        counter = col_ptr.copy()
        for row in range(rows):
            start = self.indptr[row]
            end = self.indptr[row + 1]
            for idx in range(start, end):
                col = self.indices[idx]
                pos = counter[col]
                data[pos] = self.data[idx]
                row_ind[pos] = row
                counter[col] += 1

        return CSCMatrix(data, row_ind, col_ptr, self.shape)
    
    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSRMatrix в COOMatrix.
        """
        from COO import COOMatrix
        rows, cols = self.shape
        data = []
        row_list = []
        col_list = []

        for r in range(rows):
            start = self.indptr[r]
            end = self.indptr[r + 1]
            for idx in range(start, end):
                data.append(self.data[idx])
                row_list.append(r)
                col_list.append(self.indices[idx])

        return COOMatrix(data, row_list, col_list, self.shape)
