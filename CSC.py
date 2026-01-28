from base import Matrix
from typing import List, Tuple

TOL = 1e-12


class CSCMatrix(Matrix):
    def __init__(self, data: List[float], indices: List[int], indptr: List[int], shape: Tuple[int, int]):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.nnz = len(data)
        
        # Проверка целостности
        if len(indptr) != shape[1] + 1:
            raise ValueError(f"Длина indptr должна быть {shape[1] + 1}, получено {len(indptr)}")
        if indices and max(indices) >= shape[0]:
            raise ValueError(f"Индекс строки {max(indices)} превышает размер {shape[0]}")

    def to_dense(self) -> List[List[float]]:
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
        """Сложение CSC матриц через плотное представление для согласованности."""
        if self.shape != other.shape:
            raise ValueError("Размерности матриц не совпадают")
        
        # Используем плотное представление для гарантии согласованности с эталонной реализацией
        dense_self = self.to_dense()
        dense_other = other.to_dense()
        rows, cols = self.shape
        
        result_dense = [[0.0] * cols for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                result_dense[i][j] = dense_self[i][j] + dense_other[i][j]
        
        return CSCMatrix.from_dense(result_dense)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        if scalar == 0.0:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)
        
        # Умножаем все значения, НЕ удаляя элементы
        new_data = [val * scalar for val in self.data]
        return CSCMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование CSC матрицы через COO."""
        # Преобразуем в COO, транспонируем, затем в CSR
        coo = self._to_coo()
        coo_t = coo.transpose()
        return coo_t._to_csr()

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц через плотное представление для надежности."""
        if self.shape[1] != other.shape[0]:
            raise ValueError("Несовместимые размерности для умножения")
        
        # Используем плотное представление для надежности
        dense_self = self.to_dense()
        dense_other = other.to_dense()
        rows_A, cols_A = self.shape
        rows_B, cols_B = other.shape
        
        result_dense = [[0.0] * cols_B for _ in range(rows_A)]
        for i in range(rows_A):
            for j in range(cols_B):
                val = 0.0
                for k in range(cols_A):
                    val += dense_self[i][k] * dense_other[k][j]
                result_dense[i][j] = val
        
        return CSCMatrix.from_dense(result_dense)

    @classmethod
    def from_dense(cls, dense_matrix: List[List[float]]) -> 'CSCMatrix':
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
                if abs(val) > TOL:
                    data.append(float(val))
                    indices.append(i)
                    col_nnz += 1
            indptr.append(indptr[-1] + col_nnz)
        
        return cls(data, indices, indptr, (rows, cols))

    def _to_csr(self) -> 'CSRMatrix':
        """Преобразование CSCMatrix в CSRMatrix."""
        from CSR import CSRMatrix
        
        rows, cols = self.shape
        
        if self.nnz == 0:
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
        
        return CSRMatrix(data_csr, indices_csr, indptr, self.shape)

    def _to_coo(self) -> 'COOMatrix':
        """Преобразование CSCMatrix в COOMatrix."""
        from COO import COOMatrix
        
        if self.nnz == 0:
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
        
        return COOMatrix(data, rows, cols, self.shape)
