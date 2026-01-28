from base import Matrix
from collections import defaultdict
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix

class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]

        for col in range(cols):
            start = self.indptr[col]
            end = self.indptr[col + 1]
            for idx in range(start, end):
                dense[self.indices[idx]][col] = self.data[idx]

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        if self.shape != other.shape:
            raise ValueError("матрицы разного размера")

        return self._to_coo()._add_impl(other._to_coo())._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        if scalar == 0:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)

        new_data = [val * scalar for val in self.data]

        return CSCMatrix(new_data, self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Hint:
        Результат - в CSR формате (с теми же данными, но с интерпретацией строк как столбцов).
        """
        from CSR import CSRMatrix

        rows, cols = self.shape
        data, row, col = [], [], []

        for c in range(cols):
            for idx in range(self.indptr[c], self.indptr[c + 1]):
                r = self.indices[idx]
                v = self.data[idx]
                row.append(c)  # меняем row и col
                col.append(r)
                data.append(v)

        elements = list(zip(row, col, data))
        elements.sort(key=lambda x: (x[0], x[1]))  # сортировка по row, потом col

        sorted_row = [e[0] for e in elements]
        sorted_col = [e[1] for e in elements]
        sorted_data = [e[2] for e in elements]

        new_shape = (cols, rows)
        new_cols = rows

        indptr = [0] * (new_cols + 1)
        for c in sorted_col:
            indptr[c + 1] += 1

        for i in range(1, new_cols + 1):
            indptr[i] += indptr[i - 1]

        return CSCMatrix(sorted_data, sorted_row, indptr, new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        from CSC import CSCMatrix

        if isinstance(other, CSCMatrix):
            rows_a, cols_a = self.shape
            rows_b, cols_b = other.shape

            data_res = []
            indices_res = []
            indptr_res = [0]

            for col_b in range(cols_b):
                col_vals = {}
                start_b = other.indptr[col_b]
                end_b = other.indptr[col_b + 1]

                for idx_b in range(start_b, end_b):
                    row_b = other.indices[idx_b]
                    val_b = other.data[idx_b]

                    start_a = self.indptr[row_b]
                    end_a = self.indptr[row_b + 1]
                    for idx_a in range(start_a, end_a):
                        row_a = self.indices[idx_a]
                        val_a = self.data[idx_a]
                        col_vals[row_a] = col_vals.get(row_a, 0) + val_a * val_b

                # добавляемзначения в result
                rows_in_col = sorted(col_vals.keys())
                for r in rows_in_col:
                    v = col_vals[r]
                    if v != 0:
                        data_res.append(v)
                        indices_res.append(r)
                indptr_res.append(len(data_res))

            return CSCMatrix(data_res, indices_res, indptr_res, (rows_a, cols_b))

        elif 'CSRMatrix' in str(type(other)):
            other_csc = other._to_csc()
            return self._matmul_impl(other_csc)

        elif 'COOMatrix' in str(type(other)):
            other_csc = other._to_csc()
            return self._matmul_impl(other_csc)

        #через плотную матрицу
        else:
            dense_self = self.to_dense()
            dense_other = other.to_dense()

            rows_a, cols_a = self.shape
            rows_b, cols_b = other.shape

            result = [[0.0] * cols_b for _ in range(rows_a)]
            for i in range(rows_a):
                for j in range(cols_a):
                    if dense_self[i][j] != 0:
                        for k in range(cols_b):
                            if dense_other[j][k] != 0:
                                result[i][k] += dense_self[i][j] * dense_other[j][k]

            return self.__class__.from_dense(result)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0

        data = []
        indices = []
        indptr = [0]

        for j in range(cols):
            count = 0
            for i in range(rows):
                val = dense_matrix[i][j]
                if val != 0:
                    data.append(val)
                    indices.append(i)
                    count += 1
            indptr.append(indptr[-1] + count)

        return cls(data, indices, indptr, (rows, cols))

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix

        coo = self._to_coo()
        elements = list(zip(coo.row, coo.col, coo.data))
        elements.sort(key=lambda x: (x[0], x[1]))

        sorted_row = [e[0] for e in elements]
        sorted_col = [e[1] for e in elements]
        sorted_data = [e[2] for e in elements]

        rows = self.shape[0]
        indptr = [0] * (rows + 1)
        for r in sorted_row:
            indptr[r + 1] += 1
        for i in range(1, rows + 1):
            indptr[i] += indptr[i - 1]

        return CSRMatrix(sorted_data, sorted_col, indptr, self.shape)

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        from COO import COOMatrix
        data, row, col = [], [], []
        cols = self.shape[1]
        for j in range(cols):
            for idx in range(self.indptr[j], self.indptr[j + 1]):
                row.append(self.indices[idx])
                col.append(j)
                data.append(self.data[idx])
        return COOMatrix(data, row, col, self.shape)