from base import Matrix
from types import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from COO import COOMatrix
    from CSR import CSRMatrix

ZERO_TOL = 1e-12


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        """
        Создаёт CSC-матрицу.
        data: ненулевые значения
        indices: индексы строк для каждого значения в data
        indptr: указатели на начало каждого столбца (длина cols+1)
        """
        super().__init__(shape)
        rows, cols = shape

        if len(indptr) != cols + 1:
            raise ValueError(f"Длина indptr ({len(indptr)}) должна быть cols+1 ({cols+1})")
        if indptr[-1] != len(data) or indptr[-1] != len(indices):
            raise ValueError("Несогласованность данных: indptr[-1] != len(data) или len(indices)")

        # Проверка индексов строк
        for row_idx in indices:
            if not (0 <= row_idx < rows):
                raise ValueError(f"Индекс строки {row_idx} выходит за пределы [0, {rows})")

        # Проверка корректности indptr
        for j in range(len(indptr) - 1):
            if indptr[j] > indptr[j + 1]:
                raise ValueError(f"Некорректный indptr: indptr[{j}]={indptr[j]} > indptr[{j+1}]={indptr[j+1]}")

        self.data = list(data)
        self.indices = list(indices)
        self.indptr = list(indptr)

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
        """Сложение CSC матриц через COO."""
        from COO import COOMatrix

        coo_self = self._to_coo()
        coo_other = other._to_coo() if hasattr(other, '_to_coo') else COOMatrix.from_dense(other.to_dense())
        result_coo = coo_self._add_impl(coo_other)
        return result_coo._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Скаляр должен быть числом, получен {type(scalar)}")

        if scalar == 0:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)

        new_data = [x * scalar for x in self.data]
        return CSCMatrix(new_data, self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование CSC матрицы (возвращает CSR)."""
        from COO import COOMatrix
        from CSR import CSRMatrix

        coo = self._to_coo()
        coo_t = coo.transpose()
        return coo_t._to_csr()

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц через COO."""
        from COO import COOMatrix

        coo_self = self._to_coo()
        coo_other = other._to_coo() if hasattr(other, '_to_coo') else COOMatrix.from_dense(other.to_dense())
        result_coo = coo_self._matmul_impl(coo_other)
        return result_coo

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        if not dense_matrix:
            return cls([], [], [0], (0, 0))

        rows = len(dense_matrix)
        cols = len(dense_matrix[0])
        data, indices, indptr = [], [], [0]

        for j in range(cols):
            for i in range(rows):
                val = dense_matrix[i][j]
                if abs(val) > ZERO_TOL:
                    data.append(val)
                    indices.append(i)
            indptr.append(len(data))

        return cls(data, indices, indptr, (rows, cols))

    def _to_csr(self) -> 'CSRMatrix':
        """Преобразование CSC в CSR через COO."""
        from COO import COOMatrix
        from CSR import CSRMatrix

        coo = self._to_coo()
        return coo._to_csr()

    def _to_coo(self) -> 'COOMatrix':
        """Преобразование CSC в COO."""
        from COO import COOMatrix

        rows, cols = self.shape
        data, row, col = [], [], []

        for j in range(cols):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for idx in range(start, end):
                data.append(self.data[idx])
                row.append(self.indices[idx])
                col.append(j)

        return COOMatrix(data, row, col, (rows, cols))