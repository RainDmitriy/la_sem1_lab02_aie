from base import Matrix
from types import COOData, COORows, COOCols, Shape, DenseMatrix
from collections import defaultdict

class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        if not (len(data) == len(row) == len(col)):
            raise ValueError("data, row и col должны быть одинаковой длины")

        n, m = shape
        if n < 0 or m < 0:
            raise ValueError("Размеры матрицы не могут быть отрицательными")
        
        for r in row:
            if not (0 <= r < n):
                raise ValueError(f"Индекс строки {r} вне диапазона [0, {n-1}]")
        for c in col:
            if not (0 <= c < m):
                raise ValueError(f"Индекс столбца {c} вне диапазона [0, {m-1}]")
        self.data = data
        self.rows = row
        self.cols = col
        self.nnz = len(data)

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу."""
        n, m = self.shape
        if n * m > 10000:
            raise MemoryError(f"Матрица {n}x{m} слишком большая для dense")
        
        mat = [[0.0] * m for _ in range(n)]
        for i in range(self.nnz):
            r = self.rows[i]
            c = self.cols[i]
            mat[r][c] = self.data[i]
        return mat

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц."""
        if isinstance(other, COOMatrix):
            merged = defaultdict(float)
            
            for i in range(self.nnz):
                key = (self.rows[i], self.cols[i])
                merged[key] += self.data[i]
            
            for i in range(other.nnz):
                key = (other.rows[i], other.cols[i])
                merged[key] += other.data[i]
            
            new_data, new_rows, new_cols = [], [], []
            for (r, c), val in merged.items():
                if abs(val) > 1e-12:
                    new_data.append(val)
                    new_rows.append(r)
                    new_cols.append(c)
            
            return COOMatrix(new_data, new_rows, new_cols, self.shape)

        if hasattr(other, '_to_coo'):
            return self._add_impl(other._to_coo())

        if self.shape[0] * self.shape[1] <= 10000:
            other_coo = COOMatrix.from_dense(other.to_dense())
            return self._add_impl(other_coo)
        
        raise ValueError("Нельзя складывать большие матрицы через dense")

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        new_data = [x * scalar for x in self.data]
        return COOMatrix(new_data, self.rows, self.cols, self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        new_shape = (self.shape[1], self.shape[0])
        return COOMatrix(self.data, self.cols, self.rows, new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""
        if isinstance(other, COOMatrix):
            result_dict = defaultdict(float)

            row_groups = defaultdict(list)
            for i in range(self.nnz):
                row_groups[self.rows[i]].append((self.cols[i], self.data[i]))

            col_groups = defaultdict(list)
            for i in range(other.nnz):
                col_groups[other.cols[i]].append((other.rows[i], other.data[i]))

            for i, row_items in row_groups.items():
                for k, a_val in row_items:
                    if k in col_groups:
                        for j, b_val in col_groups[k]:
                            result_dict[(i, j)] += a_val * b_val

            new_data, new_rows, new_cols = [], [], []
            for (r, c), val in result_dict.items():
                if abs(val) > 1e-12:
                    new_data.append(val)
                    new_rows.append(r)
                    new_cols.append(c)
            
            return COOMatrix(new_data, new_rows, new_cols, 
                           (self.shape[0], other.shape[1]))

        if hasattr(other, '_to_coo'):
            return self._matmul_impl(other._to_coo())

        if self.shape[0] * self.shape[1] <= 10000 and \
           other.shape[0] * other.shape[1] <= 10000:
            other_coo = COOMatrix.from_dense(other.to_dense())
            return self._matmul_impl(other_coo)
        
        raise ValueError("Нельзя умножать большие матрицы через dense")

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        
        data, row_indices, col_indices = [], [], []
        for i in range(rows):
            for j in range(cols):
                val = dense_matrix[i][j]
                if abs(val) > 1e-12:
                    data.append(val)
                    row_indices.append(i)
                    col_indices.append(j)
        
        return cls(data, row_indices, col_indices, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix
        
        if not self.data:
            _, m = self.shape
            return CSCMatrix([], [], [0] * (m + 1), self.shape)

        items = list(zip(self.cols, self.rows, self.data))
        items.sort(key=lambda x: (x[0], x[1]))
        
        data = [d for _, _, d in items]
        indices = [r for _, r, _ in items]
        _, m = self.shape
        
        indptr = [0] * (m + 1)
        for c, _, _ in items:
            indptr[c + 1] += 1
        
        for j in range(m):
            indptr[j + 1] += indptr[j]
        
        return CSCMatrix(data, indices, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix
        
        if not self.data:
            n, _ = self.shape
            return CSRMatrix([], [], [0] * (n + 1), self.shape)

        items = list(zip(self.rows, self.cols, self.data))
        items.sort(key=lambda x: (x[0], x[1]))
        
        data = [d for _, _, d in items]
        indices = [c for _, c, _ in items]
        n, _ = self.shape
        
        indptr = [0] * (n + 1)
        for r, _, _ in items:
            indptr[r + 1] += 1
        
        for i in range(n):
            indptr[i + 1] += indptr[i]
        
        return CSRMatrix(data, indices, indptr, self.shape)
    