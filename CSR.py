from base import Matrix
from types import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)

        n, m = shape
        if len(indptr) != n + 1:
            raise ValueError(f"indptr должен иметь длину n+1 = {n+1}, получено {len(indptr)}")

        for i in range(n):
            if indptr[i] > indptr[i + 1]:
                raise ValueError(f"indptr должен быть неубывающим: indptr[{i}] = {indptr[i]} > indptr[{i+1}] = {indptr[i+1]}")

        if indptr[0] != 0:
            raise ValueError(f"indptr[0] должен быть 0, получено {indptr[0]}")
        if indptr[-1] != len(data):
            raise ValueError(f"indptr[-1] должен быть равен len(data) = {len(data)}, получено {indptr[-1]}")

        if len(data) != len(indices):
            raise ValueError(f"data и indices должны быть одинаковой длины: data={len(data)}, indices={len(indices)}")

        for col_idx in indices:
            if not (0 <= col_idx < m):
                raise ValueError(f"Индекс столбца {col_idx} вне диапазона [0, {m-1}]")
            
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.nnz = len(data)

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        n, m = self.shape
        mat = [[0]*m for _ in range(n)]
        
        for i in range(n):
            start = self.indptr[i]
            end = self.indptr[i+1]
            for idx in range(start, end):
                j = self.indices[idx]
                mat[i][j] = self.data[idx]
        
        return mat

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        A = self.to_dense()
        B = other.to_dense()
        n, m = self.shape
        
        result = []
        for i in range(n):
            row = []
            for j in range(m):
                row.append(A[i][j] + B[i][j])
            result.append(row)
        
        return CSRMatrix.from_dense(result)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        new_data = [x * scalar for x in self.data]
        return CSRMatrix(new_data, self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Hint:
        Результат - в CSC формате (с теми же данными, но с интерпретацией столбцов как строк).
        """
        coo = self._to_coo()
        transposed_coo = coo.transpose()
        return transposed_coo._to_csc()

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        A = self.to_dense()
        B = other.to_dense()
        n = len(A)
        m = len(B[0]) if B else 0
        
        C = [[0]*m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                s = 0
                for k in range(len(A[0])):
                    s += A[i][k] * B[k][j]
                C[i][j] = s
        
        return CSRMatrix.from_dense(C)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        from COO import COOMatrix
        coo = COOMatrix.from_dense(dense_matrix)
        return coo._to_csr()

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSRMatrix в CSCMatrix.
        """
        return self.transpose()
    
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

        return COOMatrix(data, row_indices, col_indices, self.shape)
