from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix

class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        if len(data) != len(row) or len(data) != len(col):
            raise ValueError("Длины data, row и col должны совпадать")

        self.data = data
        self.row = row
        self.col = col

    def iter_nonzero(self):
        """Итератор по ненулевым элементам."""
        for value, i, j in zip(self.data, self.row, self.col):
            yield i, j, value        

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу."""
        rows, cols = self.shape
        if rows * cols > 10000:
            raise ValueError("Матрица слишком большая для плотного представления")
        dense = [[0.0] * cols for _ in range(rows)]
        for i, j, value in self.iter_nonzero():
            dense[i][j] += value
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц."""
        if not isinstance(other, COOMatrix):
            other = COOMatrix.from_dense(other.to_dense())

        plus_res = {}

        for i, j, v in self.iter_nonzero():
            plus_res[(i, j)] = plus_res.get((i, j), 0) + v

        for i, j, v in other.iter_nonzero():
            plus_res[(i, j)] = plus_res.get((i, j), 0) + v

        data, row, col = [], [], []
        for (i, j), v in sorted(plus_res.items()):
            if v != 0:
                data.append(v)
                row.append(i)
                col.append(j)

        return COOMatrix(data, row, col, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        if scalar == 0:
            return COOMatrix([], [], [], self.shape)
        
        new_data = [v * scalar for v in self.data]
        return COOMatrix(new_data, self.row[:], self.col[:], self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        rows, cols = self.shape
        new_shape = (cols, rows)
        
        return COOMatrix(self.data[:], self.col[:], self.row[:], new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""
        if self.shape[1] != other.shape[0]:
            raise ValueError("Несовместимые размеры")
    
        csr_self = self._to_csr()
        csr_other = other._to_csr()
        result_csr = csr_self._matmul_impl(csr_other)
        return result_csr._to_coo()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        data, row, col = [], [], []

        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0

        for i in range(rows):
            row_data = dense_matrix[i]
            for j in range(cols):
                value = row_data[j]
                if value != 0:
                    data.append(value)
                    row.append(i)
                    col.append(j)

        return cls(data, row, col, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix
        _, num_cols = self.shape

        cols_elements = [[] for _ in range(num_cols)]
        
        data_iter = self.data
        row_iter = self.row
        col_iter = self.col
        
        for idx, c in enumerate(col_iter):
            cols_elements[c].append((row_iter[idx], data_iter[idx]))

        data, row, indptr = [], [], [0]

        for c in range(num_cols):
            for r, v in cols_elements[c]:
                data.append(v)
                row.append(r)
            indptr.append(len(data))

        return CSCMatrix(data, row, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix
        num_rows, _ = self.shape

        rows_elements = [[] for _ in range(num_rows)]
        
        data_iter = self.data
        row_iter = self.row
        col_iter = self.col
        
        for idx, r in enumerate(row_iter):
            rows_elements[r].append((col_iter[idx], data_iter[idx]))

        data, indices, indptr = [], [], [0]

        for i in range(num_rows):
            row_items = sorted(rows_elements[i])
            for c, v in row_items:
                data.append(v)
                indices.append(c)
            indptr.append(len(data))

        return CSRMatrix(data, indices, indptr, self.shape)
