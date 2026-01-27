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
            for k in range(start, end):
                row = self.indices[k]
                dense[row][j] = self.data[k]
        
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        if self.shape != other.shape:
            raise ValueError("Размерности матриц не совпадают")
        
        # Преобразуем в CSR для сложения
        csr_self = self._to_csr()
        
        # Преобразуем other в CSR если нужно
        if not isinstance(other, CSRMatrix):
            if hasattr(other, '_to_csr'):
                other_csr = other._to_csr()
            else:
                other_csr = CSRMatrix.from_dense(other.to_dense())
        else:
            other_csr = other
        
        result_csr = csr_self._add_impl(other_csr)
        return result_csr._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        if scalar == 0:
            # Возвращаем нулевую матрицу
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
        return csr_self._matmul_impl(other)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        if not dense_matrix:
            return cls([], [], [0], (0, 0))
        
        rows = len(dense_matrix)
        cols = len(dense_matrix[0])
        
        # Сначала собираем все ненулевые элементы
        elements = []
        for i in range(rows):
            for j in range(cols):
                val = dense_matrix[i][j]
                if val != 0:
                    elements.append((j, i, val))  # (col, row, val)
        
        # Сортируем по столбцам, затем по строкам
        elements.sort()
        
        data = [elem[2] for elem in elements]
        indices = [elem[1] for elem in elements]
        
        # Строим indptr
        indptr = [0] * (cols + 1)
        for elem in elements:
            indptr[elem[0] + 1] += 1
        
        # Накопление
        for j in range(cols):
            indptr[j + 1] += indptr[j]
        
        return cls(data, indices, indptr, (rows, cols))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix
        
        rows, cols = self.shape
        
        if self.nnz == 0:
            return CSRMatrix([], [], [0] * (rows + 1), self.shape)
        
        # Подсчет количества ненулевых элементов в каждой строке
        row_counts = [0] * rows
        for row in self.indices:
            row_counts[row] += 1
        
        # Строим indptr для CSR
        indptr = [0] * (rows + 1)
        for i in range(rows):
            indptr[i + 1] = indptr[i] + row_counts[i]
        
        # Рабочие массивы для построения
        current_pos = indptr.copy()
        new_data = [0.0] * self.nnz
        new_indices = [0] * self.nnz
        
        # Заполняем CSR данные
        for j in range(cols):
            for k in range(self.indptr[j], self.indptr[j + 1]):
                i = self.indices[k]
                pos = current_pos[i]
                new_data[pos] = self.data[k]
                new_indices[pos] = j
                current_pos[i] += 1
        
        return CSRMatrix(new_data, new_indices, indptr, self.shape)

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        from COO import COOMatrix
        
        if self.nnz == 0:
            return COOMatrix([], [], [], self.shape)
        
        data = []
        rows = []
        cols = []
        
        for j in range(self.shape[1]):
            for k in range(self.indptr[j], self.indptr[j + 1]):
                data.append(self.data[k])
                rows.append(self.indices[k])
                cols.append(j)
        
        return COOMatrix(data, rows, cols, self.shape)
