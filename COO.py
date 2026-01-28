from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix
from CSC import CSCMatrix
from CSR import CSRMatrix


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        # копирование данных
        self.data = [float(x) for x in data]
        self.row = list(row)
        self.col = list(col)

    def to_dense(self) -> DenseMatrix:
        rows, cols = self.shape
        res = [[0.0] * cols for _ in range(rows)]
        for k in range(len(self.data)):
            res[self.row[k]][self.col[k]] = self.data[k]
        return res

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        if not isinstance(other, COOMatrix):
            other = other._to_coo() if hasattr(other, "_to_coo") else COOMatrix.from_dense(other.to_dense())

        # использование словаря для суммы
        grid = {}
        for i in range(len(self.data)):
            pos = (self.row[i], self.col[i])
            grid[pos] = grid.get(pos, 0.0) + self.data[i]

        for i in range(len(other.data)):
            pos = (other.row[i], other.col[i])
            grid[pos] = grid.get(pos, 0.0) + other.data[i]

        v_sum, r_sum, c_sum = [], [], []
        for (r, c), val in grid.items():
            if val != 0:
                v_sum.append(val);
                r_sum.append(r);
                c_sum.append(c)
        return COOMatrix(v_sum, r_sum, c_sum, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        if scalar == 0:
            return COOMatrix([], [], [], self.shape)
        return COOMatrix([val * scalar for val in self.data], self.row[:], self.col[:], self.shape)

    def transpose(self) -> 'Matrix':
        r, c = self.shape
        return COOMatrix(self.data[:], self.col[:], self.row[:], (c, r))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        if not isinstance(other, COOMatrix):
            other = other._to_coo() if hasattr(other, "_to_coo") else COOMatrix.from_dense(other.to_dense())

        # группировка правой матрицы по строкам
        lookup = {}
        for i in range(len(other.data)):
            r_idx = other.row[i]
            if r_idx not in lookup: lookup[r_idx] = []
            lookup[r_idx].append((other.col[i], other.data[i]))

        res_dict = {}
        for i in range(len(self.data)):
            c_left = self.col[i]
            if c_left in lookup:
                for c_right, v_right in lookup[c_left]:
                    target = (self.row[i], c_right)
                    res_dict[target] = res_dict.get(target, 0.0) + self.data[i] * v_right

        v_res, r_res, c_res = [], [], []
        for (r, c), val in res_dict.items():
            if val != 0:
                v_res.append(val);
                r_res.append(r);
                c_res.append(c)
        return COOMatrix(v_res, r_res, c_res, (self.shape[0], other.shape[1]))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        n = len(dense_matrix)
        m = len(dense_matrix[0]) if n > 0 else 0
        v, r, c = [], [], []
        for i in range(n):
            for j in range(m):
                if dense_matrix[i][j] != 0:
                    v.append(dense_matrix[i][j]);
                    r.append(i);
                    c.append(j)
        return cls(v, r, c, (n, m))

    def _to_csc(self) -> 'CSCMatrix':
        h, w = self.shape
        counts = [0] * w
        for col_idx in self.col: counts[col_idx] += 1

        ptr = [0] * (w + 1)
        for i in range(w): ptr[i + 1] = ptr[i] + counts[i]

        move = ptr[:-1].copy()
        v_out, i_out = [0.0] * len(self.data), [0] * len(self.data)
        for k in range(len(self.data)):
            target = move[self.col[k]]
            v_out[target], i_out[target] = self.data[k], self.row[k]
            move[self.col[k]] += 1
        return CSCMatrix(v_out, i_out, ptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        h, w = self.shape
        counts = [0] * h
        for row_idx in self.row: counts[row_idx] += 1

        ptr = [0] * (h + 1)
        for i in range(h): ptr[i + 1] = ptr[i] + counts[i]

        move = ptr[:-1].copy()
        v_out, j_out = [0.0] * len(self.data), [0] * len(self.data)
        for k in range(len(self.data)):
            target = move[self.row[k]]
            v_out[target], j_out[target] = self.data[k], self.col[k]
            move[self.row[k]] += 1
        return CSRMatrix(v_out, j_out, ptr, self.shape)