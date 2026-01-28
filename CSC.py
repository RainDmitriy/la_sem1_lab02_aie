from base import Matrix
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.shape = shape

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу."""
        n, m = self.shape
        matrix = [[0] * m for _ in range(n)]

        counter_row = 0
        col = 0
        while col < len(self.indptr) - 1:
            nnz = self.indptr[col + 1] - self.indptr[col]
            for _ in range(nnz):
                ind_row = self.indices[counter_row]
                matrix[ind_row][col] = self.data[counter_row]
                counter_row += 1
            col += 1

        return matrix

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        indptr, indices, data = [0], list(), list()
        counter_row1, counter_row2 = 0, 0
        col = 0
        while col < len(self.indptr) - 1:
            merged_row = dict()

            nnz_1 = self.indptr[col + 1] - self.indptr[col]
            for _ in range(nnz_1):
                key = self.indices[counter_row1]
                merged_row[key] = merged_row.get(key, 0) + self.data[counter_row1]
                counter_row1 += 1

            nnz_2 = other.indptr[col + 1] - other.indptr[col]
            for _ in range(nnz_2):
                key = other.indices[counter_row2]
                merged_row[key] = merged_row.get(key, 0) + other.data[counter_row2]
                counter_row2 += 1

            added_counter = 0
            if merged_row:
                for r, val in sorted(merged_row.items()):
                    if val != 0:
                        indices.append(r)
                        data.append(val)
                        added_counter += 1

            indptr.append(indptr[-1] + added_counter)

            col += 1

        return CSCMatrix(data, indices, indptr, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        if scalar == 0:
            return CSCMatrix(list(), list(), [0] * len(self.indptr), self.shape)

        new_data = [x * scalar for x in self.data]

        return CSCMatrix(new_data, self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Hint:
        Результат - в CSR формате (с теми же данными, но с интерпретацией строк как столбцов).
        """
        from CSR import CSRMatrix
        return CSRMatrix(self.data, self.indices, self.indptr, (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        csr_self = self._to_csr()
        csr_other = other._to_csr()

        return csr_self._matmul_impl(csr_other)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        indptr, indices, data = [0], list(), list()
        n, m = len(dense_matrix), len(dense_matrix[0])

        for i in range(n):
            indices_col = list()
            for j in range(m):
                elem = dense_matrix[j][i]
                if elem != 0:
                    data.append(elem)
                    indices_col.append(j)

            indptr.append(len(indices_col) + indptr[-1])
            indices.extend(indices_col)

        return CSCMatrix(data, indices, indptr, (n, m))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix

        n, m = self.shape
        nnz = len(self.data)

        row_counts = [0] * n
        for row_ind in self.indices:
            row_counts[row_ind] += 1

        csr_indptr = [0] * (n + 1)
        cumsum = 0
        for i in range(n):
            csr_indptr[i] = cumsum
            cumsum += row_counts[i]
        csr_indptr[n] = cumsum

        work_ptr = csr_indptr[:-1].copy()
        csr_indices, csr_data = [0] * nnz, [0] * nnz

        for j in range(m):
            for k in range(self.indptr[j], self.indptr[j + 1]):
                row_ind = self.indices[k]
                val = self.data[k]
                dest = work_ptr[row_ind]

                csr_indices[dest] = j
                csr_data[dest] = val

                work_ptr[row_ind] += 1

        return CSRMatrix(csr_data, csr_indices, csr_indptr, self.shape)

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        from COO import COOMatrix

        rows, cols = self.indices.copy(), list()
        m = self.shape[1]

        for j in range(m):
            nnz_col = self.indptr[j + 1] - self.indptr[j]
            cols.extend([j] * nnz_col)

        return COOMatrix(self.data, rows, cols, self.shape)
