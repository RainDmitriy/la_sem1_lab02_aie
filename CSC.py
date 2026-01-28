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
        Результат - CSR матрица.
        """
        from CSR import CSRMatrix

        rows, cols = self.shape
        nnz = len(self.data)

        if nnz == 0:
            return CSRMatrix([], [], [0] * (rows + 1), (cols, rows))

        row_counts = [0] * rows
        for j in range(cols):
            start, end = self.indptr[j], self.indptr[j + 1]
            for pos in range(start, end):
                i = self.indices[pos]
                row_counts[i] += 1

        csr_indptr = [0] * (rows + 1)
        for i in range(rows):
            csr_indptr[i + 1] = csr_indptr[i] + row_counts[i]

        csr_data = [0.0] * nnz
        csr_indices = [0] * nnz

        current_pos = csr_indptr.copy()

        for j in range(cols):
            start, end = self.indptr[j], self.indptr[j + 1]
            for pos in range(start, end):
                i = self.indices[pos]
                val = self.data[pos]

                csr_pos = current_pos[i]
                csr_data[csr_pos] = val
                csr_indices[csr_pos] = j
                current_pos[i] += 1

        coo = self._to_coo()
        transposed_coo = coo.transpose()
        return transposed_coo._to_csr()

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение матриц напрямую в разреженном формате."""
        from CSR import CSRMatrix
        from COO import COOMatrix

        coo_self = self._to_coo()
        csr_self = coo_self._to_csr()


        if isinstance(other, CSRMatrix):
            return csr_self._matmul_impl(other)
        elif isinstance(other, CSCMatrix):
            coo_other = other._to_coo()
            csr_other = coo_other._to_csr()
            return csr_self._matmul_impl(csr_other)
        elif isinstance(other, COOMatrix):
            csr_other = other._to_csr()
            return csr_self._matmul_impl(csr_other)
        else:
            other_csr = other._to_csr() if hasattr(other, '_to_csr') else CSRMatrix.from_dense(other.to_dense())
            return csr_self._matmul_impl(other_csr)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0

        coo = COOMatrix.from_dense(dense_matrix)
        return coo._to_csc()

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        return self.transpose()

    def _to_coo(self) -> 'COOMatrix':
        """Преобразование CSCMatrix в COOMatrix."""
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