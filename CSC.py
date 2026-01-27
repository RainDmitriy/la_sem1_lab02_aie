from base import Matrix
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.nnz = len(data)

    def to_dense(self) -> DenseMatrix:
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]
        
        for j in range(cols):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for k in range(start, end):
                row = self.indices[k]
                dense[row][j] = self.data[k]
        
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        # Преобразуем в COO для сложения
        from COO import COOMatrix
        
        coo_self = self._to_coo()
        coo_other = other._to_coo()
        coo_result = coo_self._add_impl(coo_other)
        return coo_result._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        if abs(scalar) < 1e-7:
            return CSCMatrix([], [], [0] * (self.cols + 1), self.shape)
        new_data = [val * scalar for val in self.data]
        return CSCMatrix(new_data, self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Возвращает CSR матрицу.
        """
        from CSR import CSRMatrix
        
        rows, cols = self.shape
        
        if self.nnz == 0:
            return CSRMatrix([], [], [0] * (rows + 1), (cols, rows))
        
        # Считаем количество элементов в каждой строке
        row_counts = [0] * rows
        for i in self.indices:
            row_counts[i] += 1
        
        # Строим indptr для CSR
        csr_indptr = [0] * (rows + 1)
        for i in range(rows):
            csr_indptr[i + 1] = csr_indptr[i] + row_counts[i]
        
        # Временный массив для текущих позиций
        temp_pos = csr_indptr[:]
        
        # Создаем массивы для CSR
        csr_data = [0.0] * self.nnz
        csr_indices = [0] * self.nnz
        
        # Заполняем CSR
        for j in range(cols):
            for k in range(self.indptr[j], self.indptr[j + 1]):
                i = self.indices[k]
                pos = temp_pos[i]
                csr_data[pos] = self.data[k]
                csr_indices[pos] = j
                temp_pos[i] += 1
        
        return CSRMatrix(csr_data, csr_indices, csr_indptr, (cols, rows))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        # Преобразуем в CSR для умножения
        csr_self = self._to_csr()
        return csr_self._matmul_impl(other)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        
        data = []
        indices = []
        indptr = [0]
        
        for j in range(cols):
            nnz_in_col = 0
            for i in range(rows):
                val = dense_matrix[i][j]
                if abs(val) > 1e-7:
                    data.append(val)
                    indices.append(i)
                    nnz_in_col += 1
            indptr.append(indptr[-1] + nnz_in_col)
        
        return cls(data, indices, indptr, (rows, cols))

    def _to_csr(self) -> 'CSRMatrix':
        from CSR import CSRMatrix
        
        rows, cols = self.shape
        
        if self.nnz == 0:
            return CSRMatrix([], [], [0] * (rows + 1), self.shape)
        
        # Считаем количество элементов в каждой строке
        row_counts = [0] * rows
        for i in self.indices:
            row_counts[i] += 1
        
        # Строим indptr для CSR
        csr_indptr = [0] * (rows + 1)
        for i in range(rows):
            csr_indptr[i + 1] = csr_indptr[i] + row_counts[i]
        
        # Временный массив для текущих позиций
        temp_pos = csr_indptr[:]
        
        # Создаем массивы для CSR
        csr_data = [0.0] * self.nnz
        csr_indices = [0] * self.nnz
        
        # Заполняем CSR
        for j in range(cols):
            for k in range(self.indptr[j], self.indptr[j + 1]):
                i = self.indices[k]
                pos = temp_pos[i]
                csr_data[pos] = self.data[k]
                csr_indices[pos] = j
                temp_pos[i] += 1
        
        return CSRMatrix(csr_data, csr_indices, csr_indptr, self.shape)

    def _to_coo(self) -> 'COOMatrix':
        from COO import COOMatrix
        
        data = self.data[:]
        rows = self.indices[:]
        cols = []
        
        for j in range(self.cols):
            for k in range(self.indptr[j], self.indptr[j + 1]):
                cols.append(j)
        
        return COOMatrix(data, rows, cols, self.shape)
