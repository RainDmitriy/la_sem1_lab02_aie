from base import Matrix
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        if len(indptr) != shape[1] + 1:
            raise ValueError("CSC: indptr должен быть длины cols + 1")
        if len(data) != len(indices):
            raise ValueError("CSC: data и indices должны быть одной длины")
        self.data = list(data)
        self.indices = list(indices)
        self.indptr = list(indptr)

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0 for _ in range(cols)] for _ in range(rows)]
        for col in range(cols):
            start = self.indptr[col]
            end = self.indptr[col + 1]
            for idx in range(start, end):
                row = self.indices[idx]
                dense[row][col] = self.data[idx]
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        if not isinstance(other, CSCMatrix):
            other = other._to_csc()
        if self.shape != other.shape:
            raise ValueError("Shapes must match")
        _, cols = self.shape
        result_data = []
        result_indices = []
        result_indptr = [0]

        for j in range(cols):
            a_entries = list(zip(
                self.indices[self.indptr[j]:self.indptr[j+1]],
                self.data[self.indptr[j]:self.indptr[j+1]]
            ))
            b_entries = list(zip(
                other.indices[other.indptr[j]:other.indptr[j+1]],
                other.data[other.indptr[j]:other.indptr[j+1]]
            ))

            a_entries.sort(key=lambda x: x[0])
            b_entries.sort(key=lambda x: x[0])

            a_pos = 0
            b_pos = 0

            while a_pos < len(a_entries) and b_pos < len(b_entries):
                a_row, a_val = a_entries[a_pos]
                b_row, b_val = b_entries[b_pos]

                if a_row == b_row:
                    val = a_val + b_val
                    if val != 0:
                        result_data.append(val)
                        result_indices.append(a_row)
                    a_pos += 1
                    b_pos += 1
                elif a_row < b_row:
                    result_data.append(a_val)
                    result_indices.append(a_row)
                    a_pos += 1
                else:
                    result_data.append(b_val)
                    result_indices.append(b_row)
                    b_pos += 1

            while a_pos < len(a_entries):
                a_row, a_val = a_entries[a_pos]
                result_data.append(a_val)
                result_indices.append(a_row)
                a_pos += 1

            while b_pos < len(b_entries):
                b_row, b_val = b_entries[b_pos]
                result_data.append(b_val)
                result_indices.append(b_row)
                b_pos += 1

            result_indptr.append(len(result_data))

        return CSCMatrix(result_data, result_indices, result_indptr, self.shape)
    
    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        if scalar == 0:
            return CSCMatrix([], [], [0]*(self.shape[1]+1), self.shape)
        data = [data * scalar for data in self.data]
        return CSCMatrix(data, self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Hint:
        Результат - в CSR формате (с теми же данными, но с интерпретацией строк как столбцов).
        """
        from CSR import CSRMatrix
        return CSRMatrix(
            self.data.copy(),
            self.indices.copy(),
            self.indptr.copy(),
            (self.shape[1], self.shape[0])
        )

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        A_csr = self._to_csr()
        B_csr = other._to_csr()
        result_csr = A_csr._matmul_impl(B_csr)
        return result_csr._to_csc()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0

        data = []
        indices = []
        indptr = [0]

        for j in range(cols):
            for i in range(rows):
                if dense_matrix[i][j] != 0:
                    data.append(dense_matrix[i][j])
                    indices.append(i)
            indptr.append(len(data))

        return cls(data, indices, indptr, (rows, cols))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix

        rows, cols = self.shape
        nnz = len(self.data)

        row_counts = [0] * rows
        for row in self.indices:
            row_counts[row] += 1

        row_ptr = [0] * (rows + 1)
        for row in range(rows):
            row_ptr[row + 1] = row_ptr[row] + row_counts[row]

        data = [0] * nnz
        col_ind = [0] * nnz
        counter = row_ptr.copy()
        for col in range(cols):
            start = self.indptr[col]
            end = self.indptr[col + 1]
            for idx in range(start, end):
                row = self.indices[idx]
                pos = counter[row]
                data[pos] = self.data[idx]
                col_ind[pos] = col
                counter[row] += 1

        return CSRMatrix(data, col_ind, row_ptr, self.shape)

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        from COO import COOMatrix

        rows, cols = self.shape
        data = []
        row_list = []
        col_list = []

        for c in range(cols):
            start = self.indptr[c]
            end = self.indptr[c + 1]
            for idx in range(start, end):
                data.append(self.data[idx])
                row_list.append(self.indices[idx])
                col_list.append(c)

        return COOMatrix(data, row_list, col_list, self.shape)
