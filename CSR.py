from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data.copy()
        self.indices = indices.copy()
        self.indptr = indptr.copy()

        if len(indptr) != shape[0] + 1:
            raise ValueError(f"Длина indptr должна быть {shape[0] + 1}, получено {len(indptr)}")

        if indptr[-1] != len(data):
            raise ValueError("Некорректный indptr")

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]

        for i in range(rows):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for j_idx in range(start, end):
                j = self.indices[j_idx]
                dense[i][j] = self.data[j_idx]

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        if isinstance(other, CSRMatrix):
            rows, cols = self.shape
            result_data = []
            result_indices = []
            result_indptr = [0]

            for i in range(rows):
                row_dict = {}

                for idx in range(self.indptr[i], self.indptr[i + 1]):
                    col = self.indices[idx]
                    row_dict[col] = self.data[idx]

                for idx in range(other.indptr[i], other.indptr[i + 1]):
                    col = other.indices[idx]
                    if col in row_dict:
                        row_dict[col] += other.data[idx]
                    else:
                        row_dict[col] = other.data[idx]

                sorted_cols = sorted(row_dict.keys())
                for col in sorted_cols:
                    val = row_dict[col]
                    if abs(val) > 1e-12:
                        result_data.append(val)
                        result_indices.append(col)

                result_indptr.append(len(result_data))

            return CSRMatrix(result_data, result_indices, result_indptr, self.shape)
        else:
            dense_self = self.to_dense()
            dense_other = other.to_dense()

            rows, cols = self.shape
            result = [[0.0] * cols for _ in range(rows)]

            for i in range(rows):
                for j in range(cols):
                    result[i][j] = dense_self[i][j] + dense_other[i][j]

            return CSRMatrix.from_dense(result)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        if scalar == 0:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)

        new_data = [val * scalar for val in self.data]
        return CSRMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Результат - в CSC формате.
        """
        from CSC import CSCMatrix

        new_shape = (self.shape[1], self.shape[0])

        dense = self.to_dense()
        transposed_dense = [[dense[j][i] for j in range(self.shape[0])] for i in range(self.shape[1])]
        return CSCMatrix.from_dense(transposed_dense)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        if isinstance(other, CSRMatrix):
            A_rows, A_cols = self.shape
            B_rows, B_cols = other.shape

            if A_cols != B_rows:
                raise ValueError("Несовместимые размерности для умножения")

            B_T = other.transpose()

            result_data = []
            result_indices = []
            result_indptr = [0]

            for i in range(A_rows):
                row_result = {}

                for a_idx in range(self.indptr[i], self.indptr[i + 1]):
                    k = self.indices[a_idx]
                    a_val = self.data[a_idx]

                    for b_idx in range(B_T.indptr[k], B_T.indptr[k + 1]):
                        j = B_T.indices[b_idx]
                        b_val = B_T.data[b_idx]

                        if j in row_result:
                            row_result[j] += a_val * b_val
                        else:
                            row_result[j] = a_val * b_val

                sorted_cols = sorted(row_result.keys())
                for j in sorted_cols:
                    val = row_result[j]
                    if abs(val) > 1e-12:
                        result_data.append(val)
                        result_indices.append(j)

                result_indptr.append(len(result_data))

            return CSRMatrix(result_data, result_indices, result_indptr, (A_rows, B_cols))
        else:
            dense_self = self.to_dense()
            dense_other = other.to_dense()

            A_rows, A_cols = self.shape
            B_rows, B_cols = other.shape
            result = [[0.0] * B_cols for _ in range(A_rows)]

            for i in range(A_rows):
                for j in range(B_cols):
                    for k in range(A_cols):
                        result[i][j] += dense_self[i][k] * dense_other[k][j]

            return CSRMatrix.from_dense(result)

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
                val = dense_matrix[i][j]
                if val != 0:
                    data.append(val)
                    indices.append(j)
            indptr.append(len(data))

        return cls(data, indices, indptr, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSRMatrix в CSCMatrix.
        """
        return self.transpose()

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSRMatrix в COOMatrix.
        """
        from COO import COOMatrix

        rows, cols = self.shape
        data = []
        row_indices = []
        col_indices = []

        for i in range(rows):
            for idx in range(self.indptr[i], self.indptr[i + 1]):
                data.append(self.data[idx])
                row_indices.append(i)
                col_indices.append(self.indices[idx])

        return COOMatrix(data, row_indices, col_indices, self.shape)