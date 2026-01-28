from base import Matrix
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix

TOLERANCE = 1e-12

class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        rows, cols = shape
        if len(indptr) != cols + 1:
            raise ValueError("indptr has wrong length")
        if len(data) != len(indices):
            raise ValueError("data and indices dont have same length")

        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу."""
        rows, cols = self.shape
        matrix = []
        for i in range(rows):
            row = []
            for j in range(cols):
                row.append(0.0)
            matrix.append(row)

        for col in range(cols):
            start = self.indptr[col]
            end = self.indptr[col + 1]
            for idx in range(start, end):
                value = self.data[idx]
                row_index = self.indices[idx]
                matrix[row_index][col] = value
        return matrix

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        if not isinstance(other, CSCMatrix):
            B = other._to_csc()
        else:
            B = other

        rows, cols = self.shape
        if (rows, cols) != B.shape:
            raise ValueError("matrix self and matrix other doesnt have same SHAPE")
        triples = []
        for col in range(cols):
            start1 = self.indptr[col]
            end1 = self.indptr[col + 1]
            for idx in range(start1, end1):
                row = self.indices[idx]
                val = self.data[idx]
                triples.append((row, col, val))
        for col in range(cols):
            start2 = B.indptr[col]
            end2 = B.indptr[col + 1]
            for idx in range(start2, end2):
                row = B.indices[idx]
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
        return coo_result._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        if scalar == 0.0:
            rows, cols = self.shape
            return CSCMatrix([], [], [0] * (cols + 1), self.shape)
        new_data = []
        new_indices = []
        new_indptr = [0]

        for col in range(self.shape[1]):
            start = self.indptr[col]
            end = self.indptr[col + 1]
            count = 0
            for idx in range(start, end):
                val = self.data[idx] * scalar
                if abs(val) > TOLERANCE:
                    new_data.append(val)
                    new_indices.append(self.indices[idx])
                    count += 1
            new_indptr.append(new_indptr[-1] + count)

        return CSCMatrix(new_data, new_indices, new_indptr, self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Hint:
        Результат - в CSR формате (с теми же данными, но с интерпретацией строк как столбцов).
        """
        from CSR import CSRMatrix
        new_shape = (self.shape[1], self.shape[0])
        return CSRMatrix(
            data=self.data[:],
            indices=self.indices[:],
            indptr=self.indptr[:],
            shape=new_shape
        )

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        self = self._to_coo()
        result = self._matmul_impl(other)
        return result._to_csc()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        if not dense_matrix:
            return cls([], [], [0], (0, 0))
        rows = len(dense_matrix)
        if rows == 0:
            cols = 0
        else:
            cols = len(dense_matrix[0])
        data, indices, indptr = [], [], [0]

        for col in range(cols):
            for row in range(rows):
                value = dense_matrix[row][col]
                if value != 0.0:
                    data.append(value)
                    indices.append(row)
            indptr.append(len(data))
        return cls(data, indices, indptr, (rows, cols))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        return self._to_coo()._to_csr()

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        from COO import COOMatrix
        rows, cols = self.shape
        data = []
        row_indices = []
        col_indices = []

        for j in range(cols):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for idx in range(start, end):
                data.append(self.data[idx])
                row_indices.append(self.indices[idx])
                col_indices.append(j)

        return COOMatrix(data, row_indices, col_indices, (rows, cols))