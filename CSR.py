# ==================== CSR.py (полная реализация) ====================
from base import Matrix
from typing import List, Tuple
import bisect

DenseMatrix = List[List[float]]
Shape = Tuple[int, int]


class CSRMatrix(Matrix):
    def __init__(self, data: List[float], indices: List[int], indptr: List[int], shape: Shape):
        super().__init__(shape)
        self.data = data.copy()
        self.indices = indices.copy()
        self.indptr = indptr.copy()

        if len(indptr) != shape[0] + 1:
            raise ValueError(f"Длина indptr должна быть {shape[0] + 1}, получено {len(indptr)}")

        if indptr[-1] != len(data):
            raise ValueError("Некорректный indptr")

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]

        for i in range(rows):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for j_idx in range(start, end):
                j = self.indices[j_idx]
                dense[i][j] = self.data[j_idx]

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        if isinstance(other, CSRMatrix):
            # Алгоритм сложения двух CSR матриц
            rows, cols = self.shape
            result_data = []
            result_indices = []
            result_indptr = [0]

            for i in range(rows):
                # Собираем элементы из обеих матриц в текущей строке
                row_elements = {}

                # Элементы из self
                for idx in range(self.indptr[i], self.indptr[i + 1]):
                    j = self.indices[idx]
                    row_elements[j] = self.data[idx]

                # Элементы из other
                if isinstance(other, CSRMatrix):
                    for idx in range(other.indptr[i], other.indptr[i + 1]):
                        j = other.indices[idx]
                        row_elements[j] = row_elements.get(j, 0) + other.data[idx]
                else:
                    dense_other = other.to_dense()
                    for j in range(cols):
                        if dense_other[i][j] != 0:
                            row_elements[j] = row_elements.get(j, 0) + dense_other[i][j]

                # Сортируем по столбцам и добавляем в результат
                sorted_cols = sorted(row_elements.keys())
                for j in sorted_cols:
                    if row_elements[j] != 0:  # Не храним нули
                        result_data.append(row_elements[j])
                        result_indices.append(j)

                result_indptr.append(len(result_data))

            return CSRMatrix(result_data, result_indices, result_indptr, self.shape)
        else:
            # Для других типов преобразуем в плотный формат
            dense_self = self.to_dense()
            dense_other = other.to_dense()

            rows, cols = self.shape
            result = [[0.0] * cols for _ in range(rows)]

            for i in range(rows):
                for j in range(cols):
                    result[i][j] = dense_self[i][j] + dense_other[i][j]

            return CSRMatrix.from_dense(result)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        if scalar == 0:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)

        new_data = [val * scalar for val in self.data]
        return CSRMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Результат - в CSC формате.
        """
        from CSC import CSCMatrix

        rows, cols = self.shape
        nnz = len(self.data)

        # Создаем массивы для CSC
        data = [0.0] * nnz
        indices = [0] * nnz
        indptr = [0] * (cols + 1)

        # Подсчитываем количество элементов в каждом столбце
        for j in self.indices:
            indptr[j + 1] += 1

        # Преобразуем в префиксную сумму
        for j in range(cols):
            indptr[j + 1] += indptr[j]

        # Заполняем данные
        current_pos = indptr.copy()

        for i in range(rows):
            for idx in range(self.indptr[i], self.indptr[i + 1]):
                j = self.indices[idx]
                pos = current_pos[j]
                data[pos] = self.data[idx]
                indices[pos] = i
                current_pos[j] += 1

        # Восстанавливаем оригинальный indptr
        for j in range(cols, 0, -1):
            indptr[j] = indptr[j - 1]
        indptr[0] = 0

        return CSCMatrix(data, indices, indptr, (cols, rows))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        if isinstance(other, CSRMatrix):
            # Алгоритм умножения CSR на CSR
            A_rows, A_cols = self.shape
            B_rows, B_cols = other.shape

            if A_cols != B_rows:
                raise ValueError("Несовместимые размерности для умножения")

            # Транспонируем B для эффективного доступа
            B_T = other.transpose()

            result_data = []
            result_indices = []
            result_indptr = [0]

            for i in range(A_rows):
                # Храним ненулевые элементы строки i результата
                row_result = {}

                # Получаем диапазон ненулевых элементов в строке i матрицы A
                for a_idx in range(self.indptr[i], self.indptr[i + 1]):
                    j = self.indices[a_idx]
                    a_val = self.data[a_idx]

                    # Получаем строку j матрицы B (столбец j транспонированной B)
                    if j < B_T.shape[0]:  # Проверка на всякий случай
                        for b_idx in range(B_T.indptr[j], B_T.indptr[j + 1]):
                            k = B_T.indices[b_idx]  # Это столбец B, строка транспонированной B
                            b_val = B_T.data[b_idx]

                            row_result[k] = row_result.get(k, 0) + a_val * b_val

                # Сортируем и добавляем ненулевые элементы
                sorted_cols = sorted(k for k in row_result if abs(row_result[k]) > 1e-10)
                for k in sorted_cols:
                    result_data.append(row_result[k])
                    result_indices.append(k)

                result_indptr.append(len(result_data))

            return CSRMatrix(result_data, result_indices, result_indptr, (A_rows, B_cols))
        else:
            # Для других типов преобразуем в плотный формат
            dense_self = self.to_dense()
            dense_other = other.to_dense()

            rows, cols = self.shape
            other_cols = other.shape[1]
            result = [[0.0] * other_cols for _ in range(rows)]

            for i in range(rows):
                for j in range(other_cols):
                    for k in range(cols):
                        result[i][j] += dense_self[i][k] * dense_other[k][j]

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
                val = dense_matrix[i][j]
                if abs(val) > 1e-10:  # Не храним очень маленькие значения
                    data.append(val)
                    indices.append(j)
            indptr.append(len(data))

        return cls(data, indices, indptr, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSRMatrix в CSCMatrix.
        """
        return self.transpose()  # Транспонирование CSR дает CSC

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
            for idx in range(self.indptr[i], self.indptr[i + 1]):
                data.append(self.data[idx])
                row_indices.append(i)
                col_indices.append(self.indices[idx])

        return COOMatrix(data, row_indices, col_indices, self.shape)