from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix

TOLERANCE = 1e-8

class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        rows, cols = shape
        if len(indptr) != rows + 1:
            raise ValueError("indptr has wrong length")

        if len(data) != len(indices):
            raise ValueError("data and indices dont have same length")

        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        rows, cols = self.shape
        matrix = []
        for i in range(rows):
            row = []
            for j in range(cols):
                row.append(0.0)
            matrix.append(row)

        for row in range(rows):
            start = self.indptr[row]
            end = self.indptr[row + 1]
            for idx in range(start, end):
                value = self.data[idx]
                col_index = self.indices[idx]
                matrix[row][col_index] = value
        return matrix

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        if not isinstance(other, CSRMatrix):
            B = other._to_csr()
        else:
            B = other

        rows, cols = self.shape
        if (rows, cols) != B.shape:
            raise ValueError("matrix self and matrix other doesnt have same shape ")
        triples = []
        for row in range(rows):
            start1 = self.indptr[row]
            end1 = self.indptr[row + 1]
            for idx in range(start1, end1):
                col = self.indices[idx]
                val = self.data[idx]
                triples.append((row, col, val))

        for row in range(rows):
            start2 = B.indptr[row]
            end2 = B.indptr[row + 1]
            for idx in range(start2, end2):
                col = B.indices[idx]
                val = B.data[idx]
                triples.append((row, col, val))

        result = {}
        for r, c, val in triples:
            key = (r, c)
            if key in result:
                result[key] += val
            else:
                result[key] = val
        coo_data = []
        coo_row = []
        coo_col = []
        for (r, c), val in result.items():
            if abs(val) > TOLERANCE:
                coo_data.append(val)
                coo_row.append(r)
                coo_col.append(c)

        from COO import COOMatrix
        coo_result = COOMatrix(coo_data, coo_row, coo_col, (rows, cols))
        return coo_result._to_csr()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        if scalar == 0.0:
            rows, cols = self.shape
            return CSRMatrix([], [], [0] * (rows + 1), self.shape)
        new_data = []
        new_indices = []
        for i in range(len(self.data)):
            val = self.data[i] * scalar
            if abs(val) > TOLERANCE:
                new_data.append(val)
                new_indices.append(self.indices[i])
        return CSRMatrix(new_data, new_indices, self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Hint:
        Результат - в CSC формате (с теми же данными, но с интерпретацией столбцов как строк).
        """
        from CSC import CSCMatrix
        new_shape = (self.shape[1], self.shape[0])
        return CSCMatrix(
            data=self.data[:],
            indices=self.indices[:],
            indptr=self.indptr[:],
            shape=new_shape
        )

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        self = self._to_coo()
        result = self._matmul_impl(other)
        return result._to_csr()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        if not dense_matrix:
            return cls([], [], [0], (0, 0))
        rows = len(dense_matrix)
        if rows == 0:
            cols = 0
        else:
            cols = len(dense_matrix[0])
        data, indices, indptr = [], [], [0]

        for row in range(rows):
            for col in range(cols):
                value = dense_matrix[row][col]
                if value != 0.0:
                    data.append(value)
                    indices.append(col)
            indptr.append(len(data))

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
        rows, cols = self.shape
        data = []
        row_indices = []
        col_indices = []

        for i in range(rows):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for idx in range(start, end):
                data.append(self.data[idx])
                row_indices.append(i)
                col_indices.append(self.indices[idx])

        return COOMatrix(data, row_indices, col_indices, (rows, cols))