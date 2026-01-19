from base import Matrix
from types import COOData, COORows, COOCols, Shape, DenseMatrix


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.row = row
        self.col = col

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу."""
        rows, cols = self.shape
        res = [[0.0 for i in range(cols)] for i in range(rows)]
        for value, r, c in zip(self.data, self.row, self.col):
            res[r][c] = value
        return res

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц."""
        other_matrix = other
        if not isinstance(other_matrix, COOMatrix):
            if hasattr(other_matrix, "_to_coo"):
                other_matrix = other_matrix._to_coo()
            else:
                other_matrix = COOMatrix.from_dense(other_matrix.to_dense())

        #ключ = (строка, столбец) значение = число
        merged = {}

        for val, r, c in zip(self.data, self.row, self.col):
            merged[(r, c)] = val

        for val, r, c in zip(other_matrix.data, other_matrix.row, other_matrix.col):
            current_val = merged.get((r, c), 0.0)
            merged[(r, c)] = current_val + val

        new_data = []
        new_row = []
        new_col = []

        for (r, c), val in merged.items():
            if val != 0: 
                new_data.append(val)
                new_row.append(r)
                new_col.append(c)

        return COOMatrix(new_data, new_row, new_col, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        new_data = [v * scalar for v in self.data]
        return COOMatrix(new_data, self.row, self.col, self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        new_shape = (self.shape[1], self.shape[0])
        return COOMatrix(self.data, self.col, self.row, new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""
        a, b = self.to_dense(), other.to_dense() 
        return self.from_dense([[sum(a[i][k] * b[k][j] for k in range(self.shape[1])) for j in range(other.shape[1])] for i in range(self.shape[0])])

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        rows_count = len(dense_matrix)
        cols_count = len(dense_matrix[0]) if rows_count > 0 else 0
        
        data, row, col = [], [], []
        for r in range(rows_count):
            for c in range(cols_count):
                val = dense_matrix[r][c]
                if val != 0:
                    data.append(val)
                    row.append(r)
                    col.append(c)
        return cls(data, row, col, (rows_count, cols_count))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix
    
        triples = sorted(zip(self.col, self.row, self.data))
        
        sorted_cols = [t[0] for t in triples]
        new_indices = [t[1] for t in triples]
        new_data = [t[2] for t in triples]
        
        n_cols = self.shape[1]
        new_indptr = [0] * (n_cols + 1)
        
        for c in sorted_cols:
            new_indptr[c + 1] += 1
            
        for i in range(n_cols):
            new_indptr[i + 1] += new_indptr[i]
            
        return CSCMatrix(new_data, new_indices, new_indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix

        triples = sorted(zip(self.row, self.col, self.data))
        
        sorted_rows = [t[0] for t in triples]
        new_indices = [t[1] for t in triples]
        new_data = [t[2] for t in triples]
        
        n_rows = self.shape[0]
        new_indptr = [0] * (n_rows + 1)
        
        for r in sorted_rows:
            new_indptr[r + 1] += 1
        
        for i in range(n_rows):
            new_indptr[i + 1] += new_indptr[i]
            
        return CSRMatrix(new_data, new_indices, new_indptr, self.shape)
