from base import Matrix
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data.copy()
        self.indices = indices.copy()
        self.indptr = indptr.copy()

        if len(indptr) != shape[1] + 1:
            raise ValueError(f"Длина indptr должна быть {shape[1] + 1}, получено {len(indptr)}")

        if indptr[-1] != len(data):
            raise ValueError("Некорректный indptr")

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]

        for j in range(cols):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for idx in range(start, end):
                i = self.indices[idx]
                dense[i][j] = self.data[idx]

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        coo_self = self._to_coo()
        if isinstance(other, CSCMatrix):
            coo_other = other._to_coo()
        else:
            from COO import COOMatrix
            coo_other = COOMatrix.from_dense(other.to_dense())

        result_coo = coo_self._add_impl(coo_other)
        return result_coo._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        if scalar == 0:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)

        new_data = [val * scalar for val in self.data]
        return CSCMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Результат - в CSR формате.
        """
        from CSR import CSRMatrix

        new_shape = (self.shape[1], self.shape[0])

        dense = self.to_dense()
        transposed_dense = [[dense[j][i] for j in range(self.shape[0])] for i in range(self.shape[1])]
        return CSRMatrix.from_dense(transposed_dense)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        csr_self = self._to_csr()
        if isinstance(other, CSCMatrix):
            csr_other = other._to_csr()
        else:
            from CSR import CSRMatrix
            csr_other = CSRMatrix.from_dense(other.to_dense())

        result_csr = csr_self._matmul_impl(csr_other)
        return result_csr._to_csc()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0

        data = []
        indices = []
        indptr = [0]

        for j in range(cols):
            for i in range(rows):
                val = dense_matrix[i][j]
                if val != 0:
                    data.append(val)
                    indices.append(i)
            indptr.append(len(data))

        return cls(data, indices, indptr, (rows, cols))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        return self.transpose()

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
            for idx in range(self.indptr[j], self.indptr[j + 1]):
                data.append(self.data[idx])
                row_indices.append(self.indices[idx])
                col_indices.append(j)

        return COOMatrix(data, row_indices, col_indices, self.shape)