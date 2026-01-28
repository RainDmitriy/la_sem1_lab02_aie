from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.row = row
        self.col = col
        self.shape = shape

    def to_dense(self) -> DenseMatrix:
        n, m = self.shape
        dense_matrix = [[0.0] * m for _ in range(n)]

        k = len(self.row)
        for i in range(k):
            col_val, row_val, val = self.col[i], self.row[i], self.data[i]
            dense_matrix[row_val][col_val] = val

        return dense_matrix

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        from COO import COOMatrix
        
        if not isinstance(other, COOMatrix):
            other_dense = other.to_dense()
            other = COOMatrix.from_dense(other_dense)
        
        all_row = self.row + other.row
        all_col = self.col + other.col
        all_val = self.data + other.data

        merged_coords = dict()
        for r, c, v in zip(all_row, all_col, all_val):
            key = (r, c)
            merged_coords[key] = merged_coords.get(key, 0.0) + v

        sum_row, sum_col, sum_val = [], [], []
        for (row, col), val in sorted(merged_coords.items()):
            if abs(val) > EPS:
                sum_row.append(row)
                sum_col.append(col)
                sum_val.append(val)

        return COOMatrix(sum_val, sum_row, sum_col, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        if abs(scalar) < EPS:
            return COOMatrix([], [], [], self.shape)

        new_data = [x * scalar for x in self.data]

        return COOMatrix(new_data, self.row, self.col, self.shape)

    def transpose(self) -> 'Matrix':
        return COOMatrix(self.data, self.col, self.row, (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        from COO import COOMatrix
        
        if not isinstance(other, COOMatrix):
            other_dense = other.to_dense()
            other = COOMatrix.from_dense(other_dense)
        
        n = self.shape[0]
        k = other.shape[1]
        
        merged_coords = dict()
        for r1, c1, v1 in zip(self.row, self.col, self.data):
            for r2, c2, v2 in zip(other.row, other.col, other.data):
                if c1 == r2:
                    key = (r1, c2)
                    merged_coords[key] = merged_coords.get(key, 0.0) + v1 * v2

        new_row, new_col, new_val = [], [], []
        for (r, c), v in merged_coords.items():
            if abs(v) > EPS:
                new_row.append(r)
                new_col.append(c)
                new_val.append(v)

        return COOMatrix(new_val, new_row, new_col, (n, k))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        n, m = len(dense_matrix), len(dense_matrix[0])
        shape = (n, m)
        rows, cols, val = [], [], []

        for i in range(n):
            for j in range(m):
                if abs(dense_matrix[i][j]) > EPS:
                    rows.append(i)
                    cols.append(j)
                    val.append(float(dense_matrix[i][j]))

        return cls(val, rows, cols, shape)

    def to_csc(self) -> 'CSCMatrix':
        from CSC import CSCMatrix

        all_to_sort = list(zip(self.row, self.col, self.data))
        all_to_sort.sort(key=lambda x: (x[1], x[0]))

        if all_to_sort:
            sorted_rows, sorted_cols, sorted_data = zip(*all_to_sort)
        else:
            sorted_rows, sorted_cols, sorted_data = [], [], []

        data = list(sorted_data)
        indices = list(sorted_rows)
        n = self.shape[1]
        indptr = [0] * (n + 1)

        for col in sorted_cols:
            indptr[col + 1] += 1

        for i in range(n):
            indptr[i + 1] += indptr[i]

        return CSCMatrix(data, indices, indptr, self.shape)

    def to_csr(self) -> 'CSRMatrix':
        from CSR import CSRMatrix

        all_to_sort = list(zip(self.row, self.col, self.data))
        all_to_sort.sort(key=lambda x: (x[0], x[1]))

        if all_to_sort:
            sorted_rows, sorted_cols, sorted_data = zip(*all_to_sort)
        else:
            sorted_rows, sorted_cols, sorted_data = [], [], []

        data = list(sorted_data)
        indices = list(sorted_cols)
        n = self.shape[0]
        indptr = [0] * (n + 1)

        for row in sorted_rows:
            indptr[row + 1] += 1

        for i in range(n):
            indptr[i + 1] += indptr[i]

        return CSRMatrix(data, indices, indptr, self.shape)
