from base import Matrix
from mytypes import COOData, COORows, COOCols, Shape, DenseMatrix
from typing import Dict, Tuple, List
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from CSC import CSCMatrix
    from CSR import CSRMatrix

class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)

        if not (len(data) == len(row) == len(col)):
            raise ValueError("data, row, col не равны")
        
        n, m = shape
        for r in row:
            if not (0 <= r < n):
                raise ValueError(f"Индекс строки {r} вне диапазона")
        for c in col:
            if not (0 <= c < m):
                raise ValueError(f"Индекс столбца {c} вне диапазона")
        
        self.data = data
        self.row = row
        self.col = col

    def to_dense(self) -> DenseMatrix:
        """Преобразует COO в плотную матрицу."""
        rows, cols = self.shape
        dense = []

        for i in range(rows):
            row = [0] * cols
            dense.append(row)

        for i in range(len(self.data)):
            dense[self.row[i]][self.col[i]] = self.data[i]
    
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц."""
        if not isinstance(other, COOMatrix):
            if hasattr(other, '_to_coo'):
                other = other._to_coo()
            else:
                other_dense = other.to_dense()
                other = COOMatrix.from_dense(other_dense)
        
        sum_dict: Dict[Tuple[int, int], float] = {}

        for i in range(len(self.data)):
            key = (self.row[i], self.col[i])
            sum_dict[key] = self.data[i]

        for i in range(len(other.data)):
            key = (other.row[i], other.col[i])
            if key in sum_dict:
                sum_dict[key] += other.data[i]
            else:
                sum_dict[key] = other.data[i]

        new_data, new_row, new_col = [], [], []
        for (row, col), value in sum_dict.items():
            if abs(value) > 1e-12:
                new_data.append(value)
                new_row.append(row)
                new_col.append(col)
        
        return COOMatrix(new_data, new_row, new_col, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение COO на скаляр."""
        if abs(scalar) < 1e-12:
            return COOMatrix([], [], [], self.shape)
        
        new_data = [elem * scalar for elem in self.data]
        return COOMatrix(new_data, self.row[:], self.col[:], self.shape)

    def transpose(self) -> 'Matrix':
        """Транспонирование COO матрицы."""
        new_shape = (self.shape[1], self.shape[0])
        return COOMatrix(self.data.copy(), self.col.copy(), self.row.copy(), new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение COO матриц."""
        if self.shape[1] != other.shape[0]:
            raise ValueError(f"неправильные размеры матриц")

        if not isinstance(other, COOMatrix):
            if hasattr(other, '_to_coo'):
                other = other._to_coo()
            else:
                other_dense = other.to_dense()
                other = COOMatrix.from_dense(other_dense)

        m, n = self.shape[0], other.shape[1]

        row_groups = defaultdict(list)
        for i in range(len(self.data)):
            row_groups[self.row[i]].append((self.col[i], self.data[i]))

        col_groups = defaultdict(list)
        for i in range(len(other.data)):
            col_groups[other.col[i]].append((other.row[i], other.data[i]))

        result_dict = defaultdict(float)
        for i, row_items in row_groups.items():
            for k, a_val in row_items:
                if k in col_groups:
                    for j, b_val in col_groups[k]:
                        result_dict[(i, j)] += a_val * b_val

        data, rows, cols = [], [], []
        for (i, j), val in result_dict.items():
            if abs(val) > 1e-12:
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
                if abs(value) > 1e-12:
                    data.append(value)
                    row.append(i)
                    col.append(j)

        return COOMatrix(data, row, col, (len(dense_matrix), len(dense_matrix[0])))

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование COOMatrix в CSCMatrix.
        """
        from CSC import CSCMatrix
        
        if not self.data:
            _, cols = self.shape
            return CSCMatrix([], [], [0] * (cols + 1), self.shape)

        elements = list(zip(self.col, self.row, self.data))
        elements.sort()
        
        data: List[float] = []
        indices: List[int] = []
        cols = self.shape[1]
        indptr: List[int] = [0] * (cols + 1)

        for col_idx, _, _ in elements:
            indptr[col_idx + 1] += 1

        for j in range(cols):
            indptr[j + 1] += indptr[j]

        data = [0] * len(self.data)
        indices = [0] * len(self.data)
        positions = indptr.copy()
        
        for col_idx, row_idx, value in elements:
            pos = positions[col_idx]
            data[pos] = value
            indices[pos] = row_idx
            positions[col_idx] += 1
        
        return CSCMatrix(data, indices, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование COOMatrix в CSRMatrix.
        """
        from CSR import CSRMatrix

        if not self.data:
            rows, _ = self.shape
            return CSRMatrix([], [], [0] * (rows + 1), self.shape)

        elements = list(zip(self.row, self.col, self.data))
        elements.sort()
        
        data: List[float] = []
        indices: List[int] = []
        rows = self.shape[0]
        indptr: List[int] = [0] * (rows + 1)

        for row_idx, _, _ in elements:
            indptr[row_idx + 1] += 1

        for i in range(rows):
            indptr[i + 1] += indptr[i]

        data = [0] * len(self.data)
        indices = [0] * len(self.data)
        positions = indptr.copy()
        
        for row_idx, col_idx, value in elements:
            pos = positions[row_idx]
            data[pos] = value
            indices[pos] = col_idx
            positions[row_idx] += 1
        
        return CSRMatrix(data, indices, indptr, self.shape)
    
    def _to_coo(self) -> 'COOMatrix':
        """Преобразование в COO (просто возвращает себя)."""
        return self