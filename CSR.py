from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.shape = shape

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        n, m = self.shape
        matrix = [[0] * m for _ in range(n)]

        counter_col = 0
        row = 0
        while row < len(self.indptr) - 1:
            nnz = self.indptr[row + 1] - self.indptr[row]
            for _ in range(nnz):
                ind_col = self.indices[counter_col]
                matrix[row][ind_col] = self.data[counter_col]
                counter_col += 1
            row += 1

        return matrix

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        indptr, indices, data = [0], list(), list()
        counter_col1, counter_col2 = 0, 0
        row = 0
        while row < len(self.indptr) - 1:
            merged_row = dict()

            nnz_1 = self.indptr[row + 1] - self.indptr[row]
            for _ in range(nnz_1):
                key = self.indices[counter_col1]
                merged_row[key] = merged_row.get(key, 0) + self.data[counter_col1]
                counter_col1 += 1

            nnz_2 = other.indptr[row + 1] - other.indptr[row]
            for _ in range(nnz_2):
                key = other.indices[counter_col2]
                merged_row[key] = merged_row.get(key, 0) + other.data[counter_col2]
                counter_col2 += 1

            if merged_row:
                indices_row, data_row = zip(*sorted(merged_row.items()))

                indices.extend(indices_row)
                data.extend(data_row)

            indptr.append(indptr[-1] + len(merged_row))

            row += 1

        return CSRMatrix(data, indices, indptr, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        if scalar == 0:
            return CSRMatrix(list(), list(), [0] * len(self.indptr), self.shape)

        new_data = [x * scalar for x in self.data]

        return CSRMatrix(new_data, self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Hint:
        Результат - в CSC формате (с теми же данными, но с интерпретацией столбцов как строк).
        """
        from CSC import CSCMatrix
        return CSCMatrix(self.data, self.indices, self.indptr, (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        n_rows_A = self.shape[0]
        n_cols_B = other.shape[1]

        indptr, indices, data = [0], list(), list()

        for i in range(n_rows_A):
            row_sum = {}

            for k in range(self.indptr[i], self.indptr[i + 1]):
                col_ind_a = self.indices[k]
                val_a = self.data[k]

                start_b = other.indptr[col_ind_a]
                end_b = other.indptr[col_ind_a + 1]

                for j in range(start_b, end_b):
                    col_ind_b = other.indices[j]
                    val_b = other.data[j]

                    mult = val_a * val_b
                    if mult != 0:
                        row_sum[col_ind_b] = row_sum.get(col_ind_b, 0) + mult

            if row_sum:
                sorted_items = sorted(row_sum.items())
                for col, val in sorted_items:
                    if val != 0:
                        indices.append(col)
                        data.append(val)

            indptr.append(len(data))

        return CSRMatrix(data, indices, indptr, (n_rows_A, n_cols_B))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        indptr, indices, data = [0], list(), list()
        n, m = len(dense_matrix), len(dense_matrix[0])

        for i in range(n):
            indices_row = list()
            for j in range(m):
                elem = dense_matrix[i][j]
                if elem != 0:
                    data.append(elem)
                    indices_row.append(j)

            indptr.append(len(indices_row) + indptr[-1])
            indices.extend(indices_row)

        return CSRMatrix(data, indices, indptr, (n, m))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSRMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix

        n, m = self.shape
        nnz = len(self.data)

        col_counts = [0] * m
        for col_ind in self.indices:
            col_counts[col_ind] += 1

        csc_indptr = [0] * (m + 1)
        cumsum = 0
        for j in range(m):
            csc_indptr[j] = cumsum
            cumsum += col_counts[j]
        csc_indptr[m] = cumsum

        work_ptr = csc_indptr[:-1].copy()
        csc_indices, csc_data = [0] * nnz, [0] * nnz

        for i in range(n):
            for k in range(self.indptr[i], self.indptr[i + 1]):
                col_ind = self.indices[k]
                val = self.data[k]
                dest = work_ptr[col_ind]

                csc_indices[dest] = i
                csc_data[dest] = val

                work_ptr[col_ind] += 1

        return CSCMatrix(csc_data, csc_indices, csc_indptr, self.shape)

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSRMatrix в COOMatrix.
        """
        from COO import COOMatrix

        cols, rows = self.indices.copy(), list()
        n = self.shape[0]

        for i in range(n):
            nnz_row = self.indptr[i + 1] - self.indptr[i]
            rows.extend([i] * nnz_row)

        return COOMatrix(self.data, rows, cols, self.shape)
