from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix
from typing import Dict, Tuple, List
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from CSC import CSCMatrix
    from CSR import CSRMatrix

class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        if not (len(data) == len(row) == len(col)):
            raise ValueError()
        self.data = data
        self.row = row
        self.col = col
        self.shape = shape

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу."""
        rows, cols = self.shape
        dense = []
        for i in range(rows):
            row_list = [0] * cols
            dense.append(row_list)
        for i in range(len(self.data)):
            dense[self.row[i]][self.col[i]] = self.data[i]
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц."""
        sum_dict: Dict[Tuple[int, int], float] = {}
        for value, r, c in zip(self.data, self.row, self.col):
            sum_dict[(r, c)] = sum_dict.get((r, c), 0.0) + value
        for value, r, c in zip(other.data, other.row, other.col):
            sum_dict[(r, c)] = sum_dict.get((r, c), 0.0) + value
        new_data, new_row, new_col = [], [], []
        for (r, c), val in sum_dict.items():
            if val != 0:
                new_data.append(val)
                new_row.append(r)
                new_col.append(c)
        return COOMatrix(new_data, new_row, new_col, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        new_data = [elem * scalar for elem in self.data]
        return COOMatrix(new_data, self.row[:], self.col[:], self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        new_shape = (self.shape[1], self.shape[0])
        return COOMatrix(self.data.copy(), self.col.copy(), self.row.copy(), new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""
        if self.shape[1] != other.shape[0]:
            raise ValueError()
        m, n = self.shape[0], other.shape[1]
        result = {}
        for idx in range(len(self.data)):
            row_a = self.row[idx]
            col_a = self.col[idx]
            val_a = self.data[idx]
            other_csr = other._to_csr()
            row_start = other_csr.indptr[col_a]
            row_end = other_csr.indptr[col_a + 1]
            for k in range(row_start, row_end):
                col_b = other_csr.indices[k]
                val_b = other_csr.data[k]
                key = (row_a, col_b)
                result[key] = result.get(key, 0.0) + val_a * val_b
        data, rows, cols = [], [], []
        for (i, j), val in result.items():
            if abs(val) > 1e-14:
                data.append(val)
                rows.append(i)
                cols.append(j)
        return COOMatrix(data, rows, cols, (m, n))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """Создание COO из плотной матрицы."""
        data, row, col = [], [], []
        for i in range(len(dense_matrix)):
            for j in range(len(dense_matrix[0])):
                value = dense_matrix[i][j]
                if value != 0:
                    data.append(value)
                    row.append(i)
                    col.append(j)
        return COOMatrix(data, row, col, (len(dense_matrix), len(dense_matrix[0])))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix
        rows, cols = self.shape
        elements = list(zip(self.col, self.row, self.data))
        elements.sort()
        data_list: List[float] = []
        indices_list: List[int] = []
        indptr_list: List[int] = [0] * (cols + 1)
        for col_idx, row_idx, value in elements:
            data_list.append(value)
            indices_list.append(row_idx)
            indptr_list[col_idx + 1] += 1
        for j in range(cols):
            indptr_list[j + 1] += indptr_list[j]
        return CSCMatrix(data_list, indices_list, indptr_list, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix
        rows, cols = self.shape
        elements = list(zip(self.row, self.col, self.data))
        elements.sort()
        data_list: List[float] = []
        indices_list: List[int] = []
        indptr_list: List[int] = [0] * (rows + 1)
        for row_idx, col_idx, value in elements:
            data_list.append(value)
            indices_list.append(col_idx)
            indptr_list[row_idx + 1] += 1
        for i in range(rows):
            indptr_list[i + 1] += indptr_list[i]
        return CSRMatrix(data_list, indices_list, indptr_list, self.shape)

