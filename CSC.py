from base import Matrix
from matrix_types import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix


# Сжатая разреженная (строки)
class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу."""
        h, w = self.shape
        res = [[0.0] * w for _ in range(h)]
        for j in range(w):
            for k in range(self.indptr[j], self.indptr[j + 1]):
                res[self.indices[k]][j] = self.data[k]
        return res

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        return (self._to_coo() + other)._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        new_d = [x * scalar for x in self.data]
        return CSCMatrix(new_d, list(self.indices), list(self.indptr), self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Hint: Результат - в CSR формате.
        """
        from CSR import CSRMatrix
        return CSRMatrix(
            list(self.data),
            list(self.indices),
            list(self.indptr),
            (self.shape[1], self.shape[0])
        )

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        return (self._to_coo() @ other)._to_csc()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        h = len(dense_matrix)
        w = len(dense_matrix[0]) if h else 0
        d, ind = [], []
        ptr = [0]

        for j in range(w):
            cnt = 0
            for i in range(h):
                v = dense_matrix[i][j]
                if v != 0:
                    d.append(v)
                    ind.append(i)
                    cnt += 1
            ptr.append(ptr[-1] + cnt)

        return cls(d, ind, ptr, (h, w))

    def _to_csr(self) -> 'CSRMatrix':
        """Преобразование CSCMatrix в CSRMatrix."""
        return self._to_coo()._to_csr()

    def _to_coo(self) -> 'COOMatrix':
        """Преобразование CSCMatrix в COOMatrix."""
        from COO import COOMatrix
        cols = []
        for j in range(self.shape[1]):
            count = self.indptr[j + 1] - self.indptr[j]
            cols.extend([j] * count)

        return COOMatrix(list(self.data), list(self.indices), cols, self.shape)