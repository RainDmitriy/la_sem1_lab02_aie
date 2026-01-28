from base import Matrix
from type1 import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix


class CSRMatrix(Matrix):
    def __init__(
        self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape
    ):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.shape = shape

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        dense: DenseMatrix = [[0] * self.shape[1] for i in range(self.shape[0])]

        for i in range(self.shape[0]):
            for k in range(self.indptr[i], self.indptr[i + 1]):
                col_idx = self.indices[k]
                val = self.data[k]
                dense[i][col_idx] = val
        return dense

    def _add_impl(self, other: "Matrix") -> "Matrix":
        """Сложение CSR матриц."""
        from COO import COOMatrix

        self_coo: COOMatrix = self._to_coo()
        other_coo: COOMatrix = other._to_coo()

        ans: COOMatrix = self_coo._add_impl(other_coo)

        return ans._to_csr()

    def _mul_impl(self, scalar: float) -> "Matrix":
        """Умножение CSR на скаляр."""
        for i in range(len(self.data)):
            self.data[i] *= scalar
        return self

    def transpose(self) -> "Matrix":
        """
        Транспонирование CSR матрицы.
        Hint:
        Результат - в CSC формате (с теми же данными, но с интерпретацией столбцов как строк).
        """
        from CSC import CSCMatrix

        transposed = CSCMatrix(self.data, self.indices, self.indptr, self.shape[::-1])
        return transposed

    def _matmul_impl(self, other: "Matrix") -> "Matrix":
        """Умножение CSR матриц."""
        rows_A, cols_A = self.shape
        rows_B, cols_B = other.shape

        res_data = []
        res_indices = []
        res_indptr = [0]

        spa = [0.0] * cols_B
        occupied = []

        for i in range(rows_A):
            for a_idx in range(self.indptr[i], self.indptr[i + 1]):
                k = self.indices[a_idx]
                val_A = self.data[a_idx]

                for b_idx in range(other.indptr[k], other.indptr[k + 1]):
                    col_B = other.indices[b_idx]
                    val_B = other.data[b_idx]

                    if spa[col_B] == 0:
                        occupied.append(col_B)
                    spa[col_B] += val_A * val_B

            occupied.sort()
            for col in occupied:
                if spa[col] != 0:
                    res_data.append(spa[col])
                    res_indices.append(col)
                    spa[col] = 0

            res_indptr.append(len(res_data))
            occupied = []

        return CSRMatrix(res_data, res_indices, res_indptr, (rows_A, cols_B))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> "CSRMatrix":
        """Создание CSR из плотной матрицы."""
        rows = len(dense_matrix)
        if rows > 0:
            cols = len(dense_matrix[0])
        else:
            cols = 0

        data = []
        indices = []
        indptr = [0]

        cumulative_count = 0
        for r in range(rows):
            for c in range(cols):
                val = dense_matrix[r][c]
                if val != 0:
                    data.append(float(val))
                    indices.append(c)
                    cumulative_count += 1
            indptr.append(cumulative_count)

        return cls(data, indices, indptr, (rows, cols))

    def _to_csc(self) -> "CSCMatrix":
        """
        Преобразование CSRMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix

        ans_data = []
        ans_indices = []
        ans_indptr = [0]
        cur_data = 0
        cur_row = 0
        elems = []
        for i in range(1, len(self.indptr)):
            for j in self.indices[self.indptr[i - 1] : self.indptr[i]]:
                cur_elem = [self.indices[cur_data], cur_row, self.data[cur_data]]
                elems.append(cur_elem)
                cur_data += 1
            cur_row += 1
        elems.sort()
        cur_col = 0
        num_cols = 0
        for i in range(len(elems)):
            ans_data.append(elems[i][2])
            ans_indices.append(elems[i][1])
        cur_elem = 0
        while cur_elem < len(elems):
            if cur_col == elems[cur_elem][0]:
                num_cols += 1
                cur_elem += 1
            else:
                ans_indptr.append(num_cols)
                cur_col += 1
        ans_indptr.append(num_cols)
        ans = CSCMatrix(ans_data, ans_indices, ans_indptr, self.shape)
        return ans

    def _to_coo(self) -> "COOMatrix":
        """
        Преобразование CSRMatrix в COOMatrix.
        """
        from COO import COOMatrix

        data, row, col = [], [], []
        for r in range(self.shape[0]):
            for j in range(self.indptr[r], self.indptr[r + 1]):
                data.append(self.data[j])
                row.append(r)
                col.append(self.indices[j])
        return COOMatrix(data, row, col, self.shape)
