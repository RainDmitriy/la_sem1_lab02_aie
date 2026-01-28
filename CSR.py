from base import Matrix
from matrix_types import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix


# Сжатая разреженная (столбцы)
class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        h, w = self.shape
        res = [[0.0] * w for _ in range(h)]
        for i in range(h):
            for k in range(self.indptr[i], self.indptr[i + 1]):
                res[i][self.indices[k]] = self.data[k]
        return res

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        return (self._to_coo() + other)._to_csr()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        new_d = [x * scalar for x in self.data]
        return CSRMatrix(new_d, list(self.indices), list(self.indptr), self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Hint: Результат - в CSC формате.
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
        return (self._to_coo() @ other)._to_csr()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        h = len(dense_matrix)
        w = len(dense_matrix[0]) if h else 0
        d, ind = [], []
        ptr = [0]

        for i in range(h):
            cnt = 0
            for j in range(w):
                v = dense_matrix[i][j]
                if v != 0:
                    d.append(v)
                    ind.append(j)
                    cnt += 1
            ptr.append(ptr[-1] + cnt)

        return cls(d, ind, ptr, (h, w))

    def _to_csc(self) -> 'CSCMatrix':
        """Преобразование CSRMatrix в CSCMatrix."""
        return self._to_coo()._to_csc()

    def _to_coo(self) -> 'COOMatrix':
        """Преобразование CSRMatrix в COOMatrix."""
        from COO import COOMatrix
        rows = []
        for i in range(self.shape[0]):
            count = self.indptr[i + 1] - self.indptr[i]
            rows.extend([i] * count)

        return COOMatrix(list(self.data), rows, list(self.indices), self.shape)