from base import Matrix
from types import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from COO import COOMatrix
    from CSC import CSCMatrix

ZERO_TOL = 1e-12


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        """
        Создаёт CSR-матрицу.
        data: ненулевые значения
        indices: индексы столбцов для каждого значения в data
        indptr: указатели на начало каждой строки (длина rows+1)
        """
        super().__init__(shape)
        rows, cols = shape

        if len(indptr) != rows + 1:
            raise ValueError(f"Длина indptr ({len(indptr)}) должна быть rows+1 ({rows+1})")
        if indptr[-1] != len(data) or indptr[-1] != len(indices):
            raise ValueError("Несогласованность данных: indptr[-1] != len(data) или len(indices)")

        # Проверка индексов столбцов
        for col_idx in indices:
            if not (0 <= col_idx < cols):
                raise ValueError(f"Индекс столбца {col_idx} выходит за пределы [0, {cols})")

        # Проверка корректности indptr
        for i in range(len(indptr) - 1):
            if indptr[i] > indptr[i + 1]:
                raise ValueError(f"Некорректный indptr: indptr[{i}]={indptr[i]} > indptr[{i+1}]={indptr[i+1]}")

        self.data = list(data)
        self.indices = list(indices)
        self.indptr = list(indptr)

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]

        for i in range(rows):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for idx in range(start, end):
                col = self.indices[idx]
                dense[i][col] = self.data[idx]

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц через COO."""
        from COO import COOMatrix

        coo_self = self._to_coo()
        coo_other = other._to_coo() if hasattr(other, '_to_coo') else COOMatrix.from_dense(other.to_dense())
        result_coo = coo_self._add_impl(coo_other)
        return result_coo._to_csr()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Скаляр должен быть числом, получен {type(scalar)}")

        if scalar == 0:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)

        new_data = [x * scalar for x in self.data]
        return CSRMatrix(new_data, self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование CSR матрицы (возвращает CSC)."""
        from COO import COOMatrix
        from CSC import CSCMatrix

        coo = self._to_coo()
        coo_t = coo.transpose()
        return coo_t._to_csc()

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц через COO."""
        from COO import COOMatrix

        coo_self = self._to_coo()
        coo_other = other._to_coo() if hasattr(other, '_to_coo') else COOMatrix.from_dense(other.to_dense())
        result_coo = coo_self._matmul_impl(coo_other)
        return result_coo

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        if not dense_matrix:
            return cls([], [], [0], (0, 0))

        rows = len(dense_matrix)
        cols = len(dense_matrix[0])
        data, indices, indptr = [], [], [0]

        for i in range(rows):
            for j in range(cols):
                val = dense_matrix[i][j]
                if abs(val) > ZERO_TOL:
                    data.append(val)
                    indices.append(j)
            indptr.append(len(data))

        return cls(data, indices, indptr, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """Преобразование CSR в CSC через COO."""
        from COO import COOMatrix
        from CSC import CSCMatrix

        coo = self._to_coo()
        return coo._to_csc()

    def _to_coo(self) -> 'COOMatrix':
        """Преобразование CSR в COO."""
        from COO import COOMatrix

        rows, cols = self.shape
        data, row, col = [], [], []

        for i in range(rows):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for idx in range(start, end):
                data.append(self.data[idx])
                row.append(i)
                col.append(self.indices[idx])

        return COOMatrix(data, row, col, (rows, cols))