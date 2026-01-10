from base import Matrix
from types import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        dense=[]
        rows,cols=self.shape

        for _ in range(rows):
            new_row = [0.0] * cols
            dense.append(new_row)

        for i in range(rows):
            start=self.indptr[i]
            end=self.indptr[i+1]

            for k in range(start,end):
                col_index = self.indices[k]
                value = self.data[k]
                dense[i][col_index] = value

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        self_coo = self._to_coo()
        other_coo = other._to_coo()
        result_coo = self_coo + other_coo
        return result_coo._to_csr()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        new_data = [x * scalar for x in self.data]
        return CSRMatrix(new_data, list(self.indices), list(self.indptr), self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Hint:
        Результат - в CSC формате (с теми же данными, но с интерпретацией столбцов как строк).
        """
        from CSC import CSCMatrix
        return CSCMatrix(
            list(self.data),
            list(self.indices),
            list(self.indptr),
            (self.shape[1], self.shape[0])
        )

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        dense_self = self.to_dense()
        dense_other = other.to_dense()

        rows_a = len(dense_self)
        cols_a = len(dense_self[0])
        cols_b = len(dense_other[0])

        result_dense = []
        for i in range(rows_a):
            new_row = []
            for j in range(cols_b):
                sum_val = 0.0
                for k in range(cols_a):
                    sum_val += dense_self[i][k] * dense_other[k][j]
                new_row.append(sum_val)
            result_dense.append(new_row)

        return CSRMatrix.from_dense(result_dense)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        data=[]
        indices=[]
        indptr=[0]

        rows=len(dense_matrix)
        cols=len(dense_matrix[0])
        cur_idx=0

        for i in range(rows):
            for j in range(cols):
                val=dense_matrix[i][j]
                if val!=0:
                    data.append(val)
                    indices.append(j)
                    cur_idx+=1
            indptr.append(cur_idx)
        return cls(data, indices, indptr, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSRMatrix в CSCMatrix.
        """
        return self._to_coo()._to_csc()
    
    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSRMatrix в COOMatrix.
        """
        from COO import COOMatrix

        coo_rows = []
        coo_cols = list(self.indices)
        coo_data = list(self.data)

        for i in range(len(self.indptr) - 1):
            count = self.indptr[i + 1] - self.indptr[i]
            for _ in range(count):
                coo_rows.append(i)

        return COOMatrix(coo_data, coo_rows, coo_cols, self.shape)