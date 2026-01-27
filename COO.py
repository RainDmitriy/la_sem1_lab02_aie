from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix

class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        # Атрибуты сохранены
        self.data = list(data)
        self.row = list(row)
        self.col = list(col)

    def to_dense(self) -> DenseMatrix:
        rows, cols = self.shape
        grid = [[0.0] * cols for _ in range(rows)]
        for i in range(len(self.data)):
            grid[self.row[i]][self.col[i]] = self.data[i]
        return grid

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        vals, r_idx, c_idx = [], [], []
        for i, row in enumerate(dense_matrix):
            for j, cell in enumerate(row):
                if cell != 0:
                    vals.append(cell)
                    r_idx.append(i)
                    c_idx.append(j)
        return cls(vals, r_idx, c_idx, (len(dense_matrix), len(dense_matrix[0]) if dense_matrix else 0))

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        # Используем хэш-таблицу для суммирования координат
        coords = {}
        # Собираем свои данные
        for r, c, v in zip(self.row, self.col, self.data):
            coords[(r, c)] = coords.get((r, c), 0.0) + v
        
        # Собираем данные другой матрицы через to_dense (универсально)
        o_dense = other.to_dense()
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if o_dense[i][j] != 0:
                    coords[(i, j)] = coords.get((i, j), 0.0) + o_dense[i][j]
        
        new_d, new_r, new_c = [], [], []
        for (r, c), v in coords.items():
            if abs(v) > 1e-15: # Фильтрация нулей важна для тестов
                new_d.append(v); new_r.append(r); new_c.append(c)
        return COOMatrix(new_d, new_r, new_c, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        if abs(scalar) < 1e-15:
            return COOMatrix([], [], [], self.shape)
        return COOMatrix([v * scalar for v in self.data], self.row, self.col, self.shape)

    def transpose(self) -> 'Matrix':
        return COOMatrix(self.data, self.col, self.row, (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        # Матричное умножение через плотный формат для стабильности
        r1, c1 = self.shape
        c2 = other.shape[1]
        a_dense = self.to_dense()
        b_dense = other.to_dense()
        res = [[0.0] * c2 for _ in range(r1)]
        
        for i in range(r1):
            for k in range(c1):
                if a_dense[i][k] != 0:
                    for j in range(c2):
                        res[i][j] += a_dense[i][k] * b_dense[k][j]
        return COOMatrix.from_dense(res)

    def _to_csc(self):
        from CSC import CSCMatrix
        return CSCMatrix.from_dense(self.to_dense())

    def _to_csr(self):
        from CSR import CSRMatrix
        return CSRMatrix.from_dense(self.to_dense())
