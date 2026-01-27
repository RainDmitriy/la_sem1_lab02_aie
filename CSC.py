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
        """Преобразует CSC в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]
        
        for j in range(cols):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for idx in range(start, end):
                row = self.indices[idx]
                dense[row][j] = self.data[idx]
        
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        if self.shape != other.shape:
            raise ValueError("Размерности матриц не совпадают")
        
        # Если other тоже CSC
        if type(other).__name__ == 'CSCMatrix':
            # Преобразуем в CSR для сложения
            csr_self = self._to_csr()
            csr_other = other._to_csr()
            result_csr = csr_self._add_impl(csr_other)
            return result_csr._to_csc()
        else:
            # Иначе преобразуем в плотные
            dense_self = self.to_dense()
            dense_other = other.to_dense()
            rows, cols = self.shape
            
            data = []
            indices = []
            indptr = [0]
            
            for j in range(cols):
                col_nnz = 0
                for i in range(rows):
                    val = dense_self[i][j] + dense_other[i][j]
                    if val != 0.0:
                        data.append(val)
                        indices.append(i)
                        col_nnz += 1
                indptr.append(indptr[-1] + col_nnz)
            
            return CSCMatrix(data, indices, indptr, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        if scalar == 0.0:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)
        
        new_data = [val * scalar for val in self.data]
        return CSCMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Результат - в CSR формате.
        """
        return self._to_csr()

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        if self.shape[1] != other.shape[0]:
            raise ValueError("Несовместимые размерности для умножения")
        
        # Преобразуем в CSR для умножения
        csr_self = self._to_csr()
        result_csr = csr_self._matmul_impl(other)
        return result_csr._to_csc()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        if not dense_matrix:
            return cls([], [], [0], (0, 0))
        
        rows = len(dense_matrix)
        cols = len(dense_matrix[0])
        
        data = []
        indices = []
        indptr = [0]
        
        for j in range(cols):
            col_nnz = 0
            for i in range(rows):
                val = dense_matrix[i][j]
                if val != 0.0:
                    data.append(float(val))
                    indices.append(i)
                    col_nnz += 1
            indptr.append(indptr[-1] + col_nnz)
        
        return cls(data, indices, indptr, (rows, cols))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        rows, cols = self.shape
        
        if self.nnz == 0:
            from CSR import CSRMatrix
            return CSRMatrix([], [], [0] * (rows + 1), self.shape)
        
        # Подсчитываем количество ненулевых элементов в каждой строке
        row_counts = [0] * rows
        for i in self.indices:
            row_counts[i] += 1
        
        # Строим indptr для CSR
        indptr = [0] * (rows + 1)
        for i in range(rows):
            indptr[i + 1] = indptr[i] + row_counts[i]
        
        # Рабочие массивы для заполнения
        current_pos = indptr.copy()
        data_csr = [0.0] * self.nnz
        indices_csr = [0] * self.nnz
        
        # Заполняем CSR
        for j in range(cols):
            start, end = self.indptr[j], self.indptr[j + 1]
            for idx in range(start, end):
                i = self.indices[idx]
                pos = current_pos[i]
                data_csr[pos] = self.data[idx]
                indices_csr[pos] = j
                current_pos[i] += 1
        
        from CSR import CSRMatrix
        return CSRMatrix(data_csr, indices_csr, indptr, self.shape)

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        if self.nnz == 0:
            from COO import COOMatrix
            return COOMatrix([], [], [], self.shape)
        
        data = []
        rows = []
        cols = []
        
        for j in range(self.shape[1]):
            start, end = self.indptr[j], self.indptr[j + 1]
            for idx in range(start, end):
                data.append(self.data[idx])
                rows.append(self.indices[idx])
                cols.append(j)
        
        from COO import COOMatrix
        return COOMatrix(data, rows, cols, self.shape)
