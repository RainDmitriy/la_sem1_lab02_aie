from base import Matrix
from type1 import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.shape = shape

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        matrix: DenseMatrix = [[0] * self.shape[1] for i in range(self.shape[0])]
        for i in range(len(self.indptr) - 1):
            for j in range(self.indptr[i], self.indptr[i+1]):
                matrix[i][self.indices[j]] = self.data[j]
        return matrix

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        data, indices, indptr = [], [], [0]
        for r in range(self.shape[0]):
            indices_row = self.indices[self.indptr[r]:self.indptr[r+1]]
            data_row = self.data[self.indptr[r]:self.indptr[r+1]]
            for j in range(other.indptr[r], other.indptr[r+1]):
                if other.indices[j] in indices_row:
                    index = indices_row.index(other.indices[j])
                    data_row[index] += other.data[j]
                else:
                    data_row.append(other.data[j])
                    indices_row.append(other.indices[j])
            indptr.append(indptr[-1] + len(indices_row))
            data += data_row
            indices += indices_row
        
        return CSRMatrix(data, indices, indptr, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        for i in self.data:
            i *= scalar
        return self

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Hint:
        Результат - в CSC формате (с теми же данными, но с интерпретацией столбцов как строк).
        """
        from CSC import CSCMatrix
        transposed = CSCMatrix(self.data, self.indices, self.indptr, self.shape[::-1])
        return transposed

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
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
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        data, indices, indptr = [], [], [0]
        shape = (len(dense_matrix), len(dense_matrix[0]))

        for r in range(shape[0]):
            count = 0
            for c in range(shape[1]):
                if dense_matrix[r][c] != 0:
                    data.append(dense_matrix[r][c])
                    indices.append(c)
            indptr.append(indptr[-1] + count)
        return cls(data, indices, indptr, shape)

    def _to_csc(self) -> 'CSCMatrix':
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
    
    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSRMatrix в COOMatrix.
        """
        from COO import COOMatrix
        data, row, col = [], [], []
        for r in range(self.shape[0]):
            for j in range(self.indptr[r], self.indptr[r+1]):
                data.append(self.data[j])
                row.append(r)
                col.append(self.indices[j])
        return COOMatrix(data, row, col, self.shape)
    