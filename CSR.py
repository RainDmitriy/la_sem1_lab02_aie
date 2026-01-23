from base import Matrix
from types1 import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix


class CSRMatrix(Matrix):
    def __init__(
        self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape
    ):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        self.denseMatrix: DenseMatrix = []
        for i in range(self.shape[0]):
            self.denseMatrix.append([0] * self.shape[1])

        cur_data = 0
        for i in range(1, len(self.indptr)):
            for j in self.indices[self.indptr[i - 1] : self.indptr[i]]:
                self.denseMatrix[i - 1][j] = self.data[cur_data]
                cur_data += 1

        return self.denseMatrix

    def _add_impl(self, other: "Matrix") -> "Matrix":
        """Сложение CSR матриц."""
        dense_ans: DenseMatrix = []
        for i in range(self.shape[0]):
            dense_ans.append([0] * self.shape[1])

        self_dense = self.to_dense()
        other_dense = other.to_dense()
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                dense_ans[i][j] = self_dense[i][j] + other_dense[i][j]

        ans: CSRMatrix = CSRMatrix.from_dense(dense_ans)
        return ans

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
        dense_ans: DenseMatrix = []
        for i in range(other.shape[1]):
            dense_ans.append([0] * self.shape[0])

        self_dense = self.to_dense()
        other_dense = other.to_dense()
        for m in range(self.shape[0]):
            for k in range(other.shape[1]):
                for n in range(self.shape[1]):
                    dense_ans[m][k] += self_dense[m][n] * other_dense[n][k]

        return self.from_dense(dense_ans)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> "CSRMatrix":
        """Создание CSR из плотной матрицы."""
        data = []
        indices = []
        indptr = [0]
        shape = [len(dense_matrix), len(dense_matrix[0])]

        ptr = 0
        for cur_row in range(len(dense_matrix)):
            for cur_col in range(len(dense_matrix[cur_row])):
                val = dense_matrix[cur_row][cur_col]
                if val != 0:
                    data.append(dense_matrix[cur_row][cur_col])
                    indices.append(cur_col)
                    ptr += 1
            indptr.append(ptr)

        return cls(data, indices, indptr, shape)

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

        ans_data = []
        ans_rows = []
        ans_cols = []

        cur_data = 0
        cur_row = 0
        elems = []
        for i in range(1, len(self.indptr)):
            for j in self.indices[self.indptr[i - 1] : self.indptr[i]]:
                cur_elem = [self.indices[cur_data], cur_row, self.data[cur_data]]
                elems.append(cur_elem)
                cur_data += 1
            cur_row += 1

        for elem in elems:
            ans_data.append(elem[2])
            ans_rows.append(elem[1])
            ans_cols.append(elem[0])

        ans = COOMatrix(ans_data, ans_rows, ans_cols, self.shape)
        return ans
