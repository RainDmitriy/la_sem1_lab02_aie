from base import Matrix
from types import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix
from CSC import CSCMatrix
from COO import COOMatrix

class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = list(data)
        self.indices = list(indices)
        self.indptr = list(indptr)

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        n, m = self.shape
        dense = [[0.0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for k in range(start, end):
                j = self.indices[k]
                dense[i][j] += self.data[k]
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        if not isinstance(other, CSRMatrix):
            if hasattr(other, "_to_csr"):
                other = other._to_csr()
            else:
                other = CSRMatrix.from_dense(other.to_dense())

        coo = self._to_coo()
        coo2 = other._to_coo()
        return coo._add_impl(coo2)._to_csr()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        if scalar == 0:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)
        data = [v * scalar for v in self.data]
        return CSRMatrix(data, list(self.indices), list(self.indptr), self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Hint:
        Результат - в CSC формате (с теми же данными, но с интерпретацией столбцов как строк).
        """
        n, m = self.shape
        return CSCMatrix(list(self.data), list(self.indices), list(self.indptr), (m, n))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        a = self._to_coo()
        if isinstance(other, COOMatrix):
            b = other
        elif hasattr(other, "_to_coo"):
            b = other._to_coo()
        else:
            b = COOMatrix.from_dense(other.to_dense())
        return a._matmul_impl(b)._to_csr()


    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        n = len(dense_matrix)
        m = len(dense_matrix[0]) if n > 0 else 0
        data, indices = [], []
        indptr = [0]
        count = 0
        for i in range(n):
            for j in range(m):
                v = dense_matrix[i][j]
                if v != 0:
                    data.append(v)
                    indices.append(j)
                    count += 1
            indptr.append(count)
        return cls(data, indices, indptr, (n, m))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSRMatrix в CSCMatrix.
        """
        return self._to_coo()._to_csc()
    
    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSRMatrix в COOMatrix.
        """
        n, m = self.shape
        row, col, data = [], [], []
        for i in range(n):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for k in range(start, end):
                row.append(i)
                col.append(self.indices[k])
                data.append(self.data[k])
        return COOMatrix(data, row, col, (n, m))