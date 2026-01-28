from base import Matrix
from type1 import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        self.shape = shape
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу."""
        dense: DenseMatrix = [[0] * self.shape[1] for i in range(self.shape[0])]

        for j in range(self.shape[1]):
            for k in range(self.indptr[j], self.indptr[j + 1]):
                row_idx = self.indices[k]
                val = self.data[k]
                dense[row_idx][j] = val
        return dense
    
    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        data, indices, indptr = [], [], [0]
        for c in range(self.shape[1]):
            indices_col = self.indices[self.indptr[c]:self.indptr[c+1]]
            data_col = self.data[self.indptr[c]:self.indptr[c+1]]
            for j in range(other.indptr[c], other.indptr[c+1]):
                if other.indices[j] in indices_col:
                    index = indices_col.index(other.indices[j])
                    data_col[index] += other.data[j]
                else:
                    data_col.append(other.data[j])
                    indices_col.append(other.indices[j])
            indptr.append(indptr[-1] + len(indices_col))
            data += data_col
            indices += indices_col

        return CSCMatrix(data, indices, indptr, self.shape)


    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        for i in range(len(self.data)):
            self.data[i] *= scalar
        return self
    
    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Hint:
        Результат - в CSR формате (с теми же данными, но с интерпретацией строк как столбцов).
        """
        from CSR import CSRMatrix
        transposed = CSRMatrix(self.data, self.indices, self.indptr, self.shape[::-1])
        return transposed

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
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
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        rows = len(dense_matrix)
        if rows > 0:
            cols = len(dense_matrix[0])
        else:
            cols = 0
        
        data = []
        indices = []
        indptr = [0]
        
        cumulative_count = 0
        for c in range(cols):
            for r in range(rows):
                val = dense_matrix[r][c]
                if val != 0:
                    data.append(float(val))
                    indices.append(r)
                    cumulative_count += 1
            indptr.append(cumulative_count)
            
        return cls(data, indices, indptr, (rows, cols))

    def _to_csr(self) -> 'CSRMatrix':
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

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        from COO import COOMatrix
        data, row, col = [], [], []
        for c in range(self.shape[1]):
            for j in range(self.indptr[c], self.indptr[c+1]):
                data.append(self.data[j])
                col.append(c)
                row.append(self.indices[j])
        return COOMatrix(data, row, col, self.shape)
