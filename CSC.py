from base import Matrix
from types import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)

        n, m = shape
        if len(indptr) != m + 1:
            raise ValueError(f"indptr должен иметь длину m+1 = {m+1}, получено {len(indptr)}")

        for j in range(m):
            if indptr[j] > indptr[j + 1]:
                raise ValueError(f"indptr должен быть неубывающим: indptr[{j}] = {indptr[j]} > indptr[{j+1}] = {indptr[j+1]}")

        if indptr[0] != 0:
            raise ValueError(f"indptr[0] должен быть 0, получено {indptr[0]}")
        if indptr[-1] != len(data):
            raise ValueError(f"indptr[-1] должен быть равен len(data) = {len(data)}, получено {indptr[-1]}")

        if len(data) != len(indices):
            raise ValueError(f"data и indices должны быть одинаковой длины: data={len(data)}, indices={len(indices)}")

        for row_idx in indices:
            if not (0 <= row_idx < n):
                raise ValueError(f"Индекс строки {row_idx} вне диапазона [0, {n-1}]")
            
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.nnz = len(data)

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу."""
        n, m = self.shape
        mat = [[0]*m for _ in range(n)]
        
        for j in range(m):
            start = self.indptr[j]
            end = self.indptr[j+1]
            for idx in range(start, end):
                i = self.indices[idx]
                mat[i][j] = self.data[idx]
        
        return mat

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        A = self.to_dense()
        B = other.to_dense()
        n, m = self.shape
        
        result = []
        for i in range(n):
            row = []
            for j in range(m):
                row.append(A[i][j] + B[i][j])
            result.append(row)
        
        return CSCMatrix.from_dense(result)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        new_data = [x * scalar for x in self.data]
        return CSCMatrix(new_data, self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Hint:
        Результат - в CSR формате (с теми же данными, но с интерпретацией строк как столбцов).
        """
        coo = self._to_coo()
        transposed = coo.transpose()
        return transposed._to_csr()

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
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
        
        return CSCMatrix.from_dense(C)


    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        from COO import COOMatrix
        coo = COOMatrix.from_dense(dense_matrix)
        return coo._to_csc()

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

        data, rows, cols = [], [], []
        _, m = self.shape
        
        for j in range(m):
            start = self.indptr[j]
            end = self.indptr[j+1]
            for idx in range(start, end):
                data.append(self.data[idx])
                rows.append(self.indices[idx])
                cols.append(j)
        
        return COOMatrix(data, rows, cols, self.shape)
