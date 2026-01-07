# ==================== CSC.py (полная реализация) ====================
from base import Matrix
from typing import List, Tuple

DenseMatrix = List[List[float]]
Shape = Tuple[int, int]


class CSCMatrix(Matrix):
    def __init__(self, data: List[float], indices: List[int], indptr: List[int], shape: Shape):
        super().__init__(shape)
        self.data = data.copy()
        self.indices = indices.copy()
        self.indptr = indptr.copy()

        if len(indptr) != shape[1] + 1:
            raise ValueError(f"Длина indptr должна быть {shape[1] + 1}, получено {len(indptr)}")

        if indptr[-1] != len(data):
            raise ValueError("Некорректный indptr")

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]

        for j in range(cols):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for idx in range(start, end):
                i = self.indices[idx]
                dense[i][j] = self.data[idx]

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        # Для простоты преобразуем в CSR, складываем и преобразуем обратно
        csr_self = self._to_csr()
        if isinstance(other, CSCMatrix):
            csr_other = other._to_csr()
        else:
            csr_other = CSRMatrix.from_dense(other.to_dense())

        result_csr = csr_self._add_impl(csr_other)
        return result_csr._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        if scalar == 0:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)

        new_data = [val * scalar for val in self.data]
        return CSCMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Результат - в CSR формате.
        """
        from CSR import CSRMatrix

        rows, cols = self.shape
        nnz = len(self.data)

        # Создаем массивы для CSR
        data = [0.0] * nnz
        indices = [0] * nnz
        indptr = [0] * (rows + 1)

        # Подсчитываем количество элементов в каждой строке
        for i in self.indices:
            indptr[i + 1] += 1

        # Преобразуем в префиксную сумму
        for i in range(rows):
            indptr[i + 1] += indptr[i]

        # Заполняем данные
        current_pos = indptr.copy()

        for j in range(cols):
            for idx in range(self.indptr[j], self.indptr[j + 1]):
                i = self.indices[idx]
                pos = current_pos[i]
                data[pos] = self.data[idx]
                indices[pos] = j
                current_pos[i] += 1

        # Восстанавливаем оригинальный indptr
        for i in range(rows, 0, -1):
            indptr[i] = indptr[i - 1]
        indptr[0] = 0

        return CSRMatrix(data, indices, indptr, (cols, rows))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        # Преобразуем в CSR для умножения
        csr_self = self._to_csr()
        if isinstance(other, CSCMatrix):
            csr_other = other._to_csr()
        else:
            csr_other = CSRMatrix.from_dense(other.to_dense())

        result_csr = csr_self._matmul_impl(csr_other)
        return result_csr._to_csc()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0

        # Сначала собираем данные по столбцам
        data = []
        indices = []
        indptr = [0]

        for j in range(cols):
            for i in range(rows):
                val = dense_matrix[i][j]
                if abs(val) > 1e-10:  # Не храним очень маленькие значения
                    data.append(val)
                    indices.append(i)
            indptr.append(len(data))

        return cls(data, indices, indptr, (rows, cols))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        return self.transpose()  # Транспонирование CSC дает CSR

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        from COO import COOMatrix

        rows, cols = self.shape
        data = []
        row_indices = []
        col_indices = []

        for j in range(cols):
            for idx in range(self.indptr[j], self.indptr[j + 1]):
                data.append(self.data[idx])
                row_indices.append(self.indices[idx])
                col_indices.append(j)

        return COOMatrix(data, row_indices, col_indices, self.shape)