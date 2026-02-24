from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        if len(indptr) != shape[0] + 1:
            raise ValueError("CSR: indptr должен быть длины rows + 1")
        if len(data) != len(indices):
            raise ValueError("CSR: data и indices должны быть одной длины")
        self.data = list(data)
        self.indices = list(indices)
        self.indptr = list(indptr) 

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0 for _ in range(cols)] for _ in range(rows)]
        for row in range(rows):
            start = self.indptr[row]
            end = self.indptr[row + 1]
            for idx in range(start, end):
                col = self.indices[idx]
                dense[row][col] = self.data[idx]
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        A = self.to_dense()
        B = other.to_dense()
        rows, cols = self.shape
        result = [[A[i][j] + B[i][j] for j in range(cols)] for i in range(rows)]
        return CSRMatrix.from_dense(result)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        if scalar == 0:
            return CSRMatrix([], [], [0]*(self.shape[0]+1), self.shape)
        data = [v * scalar for v in self.data]
        return CSRMatrix(data, self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Hint:
        Результат - в CSC формате (с теми же данными, но с интерпретацией столбцов как строк).
        """
        from CSC import CSCMatrix
        rows, cols = self.shape
        nnz = len(self.data)
        col_counts = [0] * cols
        for col in self.indices:
            col_counts[col] += 1
        col_ptr = [0] * (cols + 1)
        for i in range(cols):
            col_ptr[i + 1] = col_ptr[i] + col_counts[i]
        data = [0] * nnz
        row_ind = [0] * nnz
        counter = col_ptr.copy()
        for row in range(rows):
            start = self.indptr[row]
            end = self.indptr[row + 1]
            for idx in range(start, end):
                col = self.indices[idx]
                pos = counter[col]
                data[pos] = self.data[idx]
                row_ind[pos] = row
                counter[col] += 1
        return CSCMatrix(data, row_ind, col_ptr, (cols, rows))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        A = self.to_dense()
        B = other.to_dense()
        n, m = self.shape
        _, p = other.shape
        result = [[0 for _ in range(p)] for _ in range(n)]
        for i in range(n):
            for k in range(m):
                if A[i][k] != 0:
                    for j in range(p):
                        if B[k][j] != 0:
                            result[i][j] += A[i][k] * B[k][j]
        return CSRMatrix.from_dense(result)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        data = []
        indices = []
        indptr = [0]
        for i in range(rows):
            for j in range(cols):
                if dense_matrix[i][j] != 0:
                    data.append(dense_matrix[i][j])
                    indices.append(j)
            indptr.append(len(data))
        return cls(data, indices, indptr, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSRMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix
        rows, cols = self.shape
        nnz = len(self.data)
        col_counts = [0] * cols
        for col in self.indices:
            col_counts[col] += 1
        col_ptr = [0] * (cols + 1)
        for col in range(cols):
            col_ptr[col + 1] = col_ptr[col] + col_counts[col]
        data = [0] * nnz
        row_ind = [0] * nnz
        counter = col_ptr.copy()
        for row in range(rows):
            start = self.indptr[row]
            end = self.indptr[row + 1]
            for idx in range(start, end):
                col = self.indices[idx]
                pos = counter[col]
                data[pos] = self.data[idx]
                row_ind[pos] = row
                counter[col] += 1

        return CSCMatrix(data, row_ind, col_ptr, self.shape)
    
    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSRMatrix в COOMatrix.
        """
        from COO import COOMatrix
        rows, cols = self.shape
        data = []
        row = []
        col = []
        for row in range(rows):
            start = self.indptr[row]
            end = self.indptr[row + 1]
            for idx in range(start, end):
                data.append(self.data[idx])
                row.append(row)
                col.append(self.indices[idx])
        return COOMatrix(data, row, col, self.shape)
