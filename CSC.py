from base import Matrix
from type1 import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix
from base import TransposeDense


class CSCMatrix(Matrix):
    def __init__(
        self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape
    ):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу."""
        dense: DenseMatrix = []
        for i in range(self.shape[1]):
            dense.append([0] * self.shape[0])

        cur_data = 0
        for i in range(1, len(self.indptr)):
            for j in self.indices[self.indptr[i - 1] : self.indptr[i]]:
                dense[i - 1][j] = self.data[cur_data]
                cur_data += 1

        return dense

    def _add_impl(self, other: "Matrix") -> "Matrix":
        """Сложение CSC матриц."""
        dense_ans: DenseMatrix = []
        for i in range(self.shape[0]):
            dense_ans.append([0] * self.shape[1])

        self_dense = self.to_dense()
        other_dense = other.to_dense()
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                dense_ans[i][j] = self_dense[i][j] + other_dense[i][j]

        ans: CSCMatrix = CSCMatrix.from_dense(dense_ans)
        return ans

    def _mul_impl(self, scalar: float) -> "Matrix":
        """Умножение CSC на скаляр."""
        new_data = [val * scalar for val in self.data]
        return CSCMatrix(new_data, self.indices, self.indptr, self.shape)

    def transpose(self) -> "Matrix":
        """
        Транспонирование CSC матрицы.
        Hint:
        Результат - в CSR формате (с теми же данными, но с интерпретацией строк как столбцов).
        """
        from CSR import CSRMatrix

        transposed = CSRMatrix(self.data, self.indices, self.indptr, self.shape[::-1])
        return transposed

    def _matmul_impl(self, other: "Matrix") -> "Matrix":
        """Умножение CSC матриц."""
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
    def from_dense(cls, dense_matrix: DenseMatrix) -> "CSCMatrix":
        """Создание CSC из плотной матрицы."""
        data = []
        indices = []
        indptr = [0]
        shape = [len(dense_matrix), len(dense_matrix[0])]

        dense_matrix = TransposeDense(dense_matrix)
        ptr = 0
        for cur_row in range(len(dense_matrix)):
            for cur_col in range(len(dense_matrix[cur_row])):
                val = dense_matrix[cur_row][cur_col]
                if val != 0:
                    data.append(val)
                    indices.append(cur_col)
                    ptr += 1
            indptr.append(ptr)

        return cls(data, indices, indptr, shape)

    def _to_csr(self) -> "CSRMatrix":
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix

        ans_data = []
        ans_indices = []
        ans_indptr = [0]

        cur_data = 0
        cur_col = 0
        elems = []
        for i in range(1, len(self.indptr)):
            for j in self.indices[self.indptr[i - 1] : self.indptr[i]]:
                cur_elem = [self.indices[cur_data], cur_col, self.data[cur_data]]
                elems.append(cur_elem)
                cur_data += 1
            cur_col += 1

        elems.sort()
        cur_row = 0
        num_cols = 0
        for i in range(len(elems)):
            ans_data.append(elems[i][2])
            ans_indices.append(elems[i][1])

        cur_elem = 0
        while cur_elem < len(elems):
            if cur_row == elems[cur_elem][0]:
                num_cols += 1
                cur_elem += 1
            else:
                ans_indptr.append(num_cols)
                cur_row += 1

        ans_indptr.append(num_cols)

        ans = CSRMatrix(ans_data, ans_indices, ans_indptr, self.shape)
        return ans

    def _to_coo(self) -> "COOMatrix":
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        from COO import COOMatrix

        ans_data = []
        ans_rows = []
        ans_cols = []

        cur_data = 0
        cur_col = 0
        elems = []
        for i in range(1, len(self.indptr)):
            for j in self.indices[self.indptr[i - 1] : self.indptr[i]]:
                cur_elem = [self.indices[cur_data], cur_col, self.data[cur_data]]
                elems.append(cur_elem)
                cur_data += 1
            cur_col += 1

        for elem in elems:
            ans_data.append(elem[2])
            ans_rows.append(elem[0])
            ans_cols.append(elem[1])

        ans = COOMatrix(ans_data, ans_rows, ans_cols, self.shape)
        return ans

