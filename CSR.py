from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from CSR import CSRMatrix
    from CSC import CSCMatrix
    from COO import COOMatrix

class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        if len(indptr) != shape[0] + 1:
            raise ValueError(f"indptr должен иметь длину {shape[0] + 1}")
        if indptr[0] != 0 or indptr[-1] != len(data):
            raise ValueError("Некорректный indptr")
        if len(data) != len(indices):
            raise ValueError("Длины data и indices должны совпадать")
        self.data, self.indices, self.indptr  = data, indices, indptr

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        rows, cols = self.shape
        dense = [[0.0 for i in range(cols)] for i in range(rows)]
        for i in range(rows):
            for j in range(self.indptr[i], self.indptr[i + 1]):
                dense[i][self.indices[j]] = self.data[j]
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        self_coo = self._to_coo()
        other_coo = other._to_coo()
        result_coo = self_coo._add_impl(other_coo)
        return result_coo._to_csr()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        n = len(self.data)
        new_data = [0.0] * n
        for i in range(n):
            new_data[i] = self.data[i] * scalar
        return CSRMatrix(new_data, self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Hint:
        Результат - в CSC формате (с теми же данными, но с интерпретацией столбцов как строк).
        """
        return CSCMatrix(self.data[:], self.indices[:], self.indptr[:], (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        other_csc = other._to_csc()
        rows = self.shape[0]
        cols = other_csc.shape[1]

        res_r, res_c, res_v = [], [], []

        for i in range(rows):
            row_start = self.indptr[i]
            row_end = self.indptr[i + 1]

            for j in range(cols):
                col_start = other_csc.indptr[j]
                col_end = other_csc.indptr[j + 1]

                s = 0.0
                k1 = row_start
                k2 = col_start

                while k1 < row_end and k2 < col_end:
                    if self.indices[k1] == other_csc.indices[k2]:
                        s += self.data[k1] * other_csc.data[k2]
                        k1 += 1
                        k2 += 1
                    elif self.indices[k1] < other_csc.indices[k2]:
                        k1 += 1
                    else:
                        k2 += 1

                if abs(s) > 1e-10:
                    res_r.append(i)
                    res_c.append(j)
                    res_v.append(s)
        from COO import COOMatrix
        coo_result = COOMatrix(res_v, res_r, res_c, (rows, cols))
        return coo_result._to_csr()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        csr_data, csr_indices, csr_indptr = [], [], [0]
        for i in range(rows):
            row_nz = 0
            for j in range(cols):
                val = dense_matrix[i][j]
                if abs(val) > 1e-10:
                    csr_indices.append(j)
                    csr_data.append(val)
                    row_nz += 1
            csr_indptr.append(csr_indptr[-1] + row_nz)
        return cls(csr_data, csr_indices, csr_indptr, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSRMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix
        coo = self._to_coo()
        temp_cols = [[] for _ in range(self.shape[1])]
        for i in range(len(coo.data)):
            temp_cols[coo.col[i]].append((coo.row[i], coo.data[i]))
        csc_data, csc_indices, csc_indptr = [], [], [0]
        for c in range(self.shape[1]):
            temp_cols[c].sort()
            for r, val in temp_cols[c]:
                csc_indices.append(r)
                csc_data.append(val)
            csc_indptr.append(len(csc_data))
        return CSCMatrix(csc_data, csc_indices, csc_indptr, self.shape)
    
    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSRMatrix в COOMatrix.
        """
        new_row = []
        new_col = []
        new_data = []

        for i in range(self.shape[0]):
            row_start = self.indptr[i]
            row_end = self.indptr[i + 1]
            for k in range(row_start, row_end):
                new_row.append(i)
                new_col.append(self.indices[k])
                new_data.append(self.data[k])

        from COO import COOMatrix
        return COOMatrix(new_data, new_row, new_col, self.shape)