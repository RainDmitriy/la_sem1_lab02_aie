from base import Matrix
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix, COOData, COORows, COOCols
from COO import COOMatrix
from collections import defaultdict


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr

        if len(indptr) != shape[1] + 1:
            raise ValueError(f"indptr должен иметь длину shape[1] + 1 = {shape[1] + 1}")

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]

        for j in range(cols):
            start, end = self.indptr[j], self.indptr[j + 1]
            for pos in range(start, end):
                i = self.indices[pos]
                val = self.data[pos]
                dense[i][j] = val

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц напрямую в разреженном формате."""
        if not isinstance(other, CSCMatrix):
            other_csc = other._to_csc() if hasattr(other, '_to_csc') else CSCMatrix.from_dense(other.to_dense())
        else:
            other_csc = other

        rows, cols = self.shape
        result_data = []
        result_indices = []
        result_indptr = [0]

        for j in range(cols):
            col_dict = defaultdict(float)

            start1, end1 = self.indptr[j], self.indptr[j + 1]
            for pos in range(start1, end1):
                row = self.indices[pos]
                col_dict[row] += self.data[pos]

            start2, end2 = other_csc.indptr[j], other_csc.indptr[j + 1]
            for pos in range(start2, end2):
                row = other_csc.indices[pos]
                col_dict[row] += other_csc.data[pos]

            sorted_rows = sorted(col_dict.keys())
            for row in sorted_rows:
                val = col_dict[row]
                if abs(val) > 1e-12:
                    result_data.append(val)
                    result_indices.append(row)

            result_indptr.append(len(result_data))

        return CSCMatrix(result_data, result_indices, result_indptr, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        new_data = [val * scalar for val in self.data]
        return CSCMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Результат - в CSR формате.
        """

        from CSR import CSRMatrix

        transposed_shape = (self.shape[1], self.shape[0])
        return CSRMatrix(
            data=self.data.copy(),
            indices=self.indices.copy(),
            indptr=self.indptr.copy(),
            shape=transposed_shape
        )

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение матриц напрямую в разреженном формате."""
        # Для умножения лучше преобразовать в CSR
        csr_self = self.transpose().transpose()  # Получаем CSR через двойное транспонирование
        return csr_self._matmul_impl(other)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0

        data = []
        indices = []
        col_indices = []

        for j in range(cols):
            for i in range(rows):
                val = dense_matrix[i][j]
                if abs(val) > 1e-12:
                    data.append(val)
                    indices.append(i)
                    col_indices.append(j)

        if not data:
            return cls([], [], [0] * (cols + 1), (rows, cols))

        col_counts = [0] * cols
        for j in col_indices:
            col_counts[j] += 1

        indptr = [0] * (cols + 1)
        for j in range(cols):
            indptr[j + 1] = indptr[j] + col_counts[j]

        temp_data = [0.0] * len(data)
        temp_indices = [0] * len(indices)
        next_pos = indptr.copy()

        sorted_indices = sorted(range(len(data)), key=lambda idx: (col_indices[idx], indices[idx]))

        for idx in sorted_indices:
            col = col_indices[idx]
            pos = next_pos[col]
            temp_data[pos] = data[idx]
            temp_indices[pos] = indices[idx]
            next_pos[col] += 1

        return cls(temp_data, temp_indices, indptr, (rows, cols))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        return self.transpose().transpose()

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        rows, cols = self.shape
        data = []
        row_indices = []
        col_indices = []

        for j in range(cols):
            start, end = self.indptr[j], self.indptr[j + 1]
            for pos in range(start, end):
                data.append(self.data[pos])
                row_indices.append(self.indices[pos])
                col_indices.append(j)

        return COOMatrix(data, row_indices, col_indices, self.shape)

    @classmethod
    def from_coo(cls, data: COOData, rows: COORows, cols: COOCols, shape: Shape) -> 'CSCMatrix':
        """Создание CSC из COO данных."""
        coo = COOMatrix(data, rows, cols, shape)
        return coo._to_csc()